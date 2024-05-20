import datetime
import gc
from pathlib import Path
import torch
from tqdm import tqdm
import wandb

from scheduler import get_scheduler
from utils import tensor2np
from vis import save_debug_images


def get_tools(loader, model, config):
    optimizer = torch.optim.AdamW(
        model.model.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )
    scheduler = get_scheduler(config, optimizer, loader)
    return (optimizer, scheduler)


def sanity_check(loader, model, device):
    model.model.eval()
    print("Sanity check:")
    for i, (images, masks) in enumerate(loader):
        print("\timages.shape", images.shape)
        print("\tmasks.shape", masks.shape)
        images = images.to(device)
        masks = masks.to(device)
        with torch.inference_mode():
            loss = model.process_batch(images, masks, "val")
        print(f"\tLoss: {loss}")
        break


def train_epoch(
    loader,
    model,
    optimizer,
    config,
    device,
    scheduler=None,
    log_loss=False,
    log_training_images=False,
):

    # Track the average loss in a tqdm progress bar
    total_loss = 0
    batch_bar = tqdm(
        total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Train"
    )

    # Reset this place to save evaluation stats
    model.outputs["train"] = []

    # Set model into training mode
    model.model.train()

    for i, (images, masks) in enumerate(loader):

        # Zero gradients (necessary to call explicitly in case you have split
        # training up across multiple devices)
        optimizer.zero_grad()

        # Run the batch through the model
        images = images.to(device)
        masks = masks.to(device)
        loss = model.process_batch(images, masks, "train")

        # Apply the gradients, update the optimizer
        loss.backward()
        optimizer.step()

        # Track the average loss in the tqdm progress bar
        total_loss += float(tensor2np(loss))
        batch_bar.set_postfix(
            loss=f"{total_loss/(i+1):.4f}", lr=f"{optimizer.param_groups[0]['lr']}"
        )
        batch_bar.update()

        # In specific scheduler circumstances (LRTest, basically) track loss
        # and LR much more closely
        if log_loss and config["wandb"] and (i % config["train_report_iter"] == 0):
            wandb.log(
                {
                    "batch-loss": loss.item(),
                    "batch-lr": float(scheduler.get_last_lr()[0]),
                }
            )

        # For some schedulers, we want to step every training batch
        if scheduler is not None:
            scheduler.step()

        # Save augmented training images for inspection
        if log_training_images:
            debug_impaths = save_debug_images(
                Path("/tmp/"),
                torch_imgs=images,
                torch_masks=masks,
                prefix=f"imgvis_{Path(config['train_augmentation_path']).stem}_batch{i:04}_",
            )
            print(f"Saved debug images: {debug_impaths}")

        del images, masks, loss
        torch.cuda.empty_cache()

    batch_bar.close()

    # Report the epoch results to wandb
    model.record_epoch_end("train", lr=float(optimizer.param_groups[0]["lr"]))

    # Return the average loss over all batches
    avg_loss = total_loss / len(loader)
    print(
        f"{str(datetime.datetime.now())}"
        f"    Avg Train Loss: {avg_loss:.4f}"
        f"    LR: {float(optimizer.param_groups[0]['lr']):.1E}"
    )
    return avg_loss


def evaluate(loader, model, device):

    # Track the average loss in a tqdm progress bar
    total_loss = 0
    batch_bar = tqdm(
        total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Val"
    )

    # Reset this place to save evaluation outputs
    model.outputs["val"] = []

    # Set model into training mode
    model.model.eval()

    for i, (images, masks) in enumerate(loader):

        # Run the batch through the model
        images = images.to(device)
        masks = masks.to(device)
        with torch.inference_mode():
            loss = model.process_batch(images, masks, "val")

        # Track the average loss in a tqdm progress bar
        total_loss += float(tensor2np(loss))
        batch_bar.set_postfix(avg_loss=f"{total_loss/(i+1):.4f}")
        batch_bar.update()

        del images, masks, loss
        torch.cuda.empty_cache()

    batch_bar.close()

    # Report the epoch results to wandb
    model.record_epoch_end("val")

    # Get the average total_loss across the epoch
    total_loss /= len(loader)
    return total_loss


def run_train(loaders, model, config, device, run, debug=False):

    train_loader, val_loader, _ = loaders

    print(f"Starting training {str(datetime.datetime.now())}")
    torch.cuda.empty_cache()
    gc.collect()

    optimizer, scheduler = get_tools(train_loader, model, config)
    step_kwargs = {
        "loader": train_loader,
        "model": model,
        "optimizer": optimizer,
        "config": config,
        "device": device,
        "log_training_images": config["log_training_images"],
    }
    if config["scheduler"] in ["CosMulti", "LRTest", "OneCycleLR"]:
        step_kwargs["scheduler"] = scheduler
        if config["scheduler"] == "LRTest":
            step_kwargs["log_loss"] = True

    if debug:
        sanity_check(val_loader, model, device)

    best_val_loss = 1e6

    for epoch in range(config["epochs"]):
        print("Epoch", epoch + 1)

        # Do the training epoch
        avg_train_loss = train_epoch(**step_kwargs)

        # Handle a specific scheduler
        if config["scheduler"] == "StepLR":
            scheduler.step()

        # Do the validation epoch if conditions are right
        if epoch % config["eval_report_iter"] == 0 or epoch == config["epochs"] - 1:
            avg_val_loss = evaluate(val_loader, model, device)
            print(
                f"{str(datetime.datetime.now())}    Validation Loss: {avg_val_loss:.4f}"
            )
            if config["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(avg_val_loss)

        # Save the best current model
        if avg_val_loss < best_val_loss:
            print("Saving a new best model")
            if config["wandb"]:
                torch.save(
                    {
                        "model_state_dict": model.model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "avg_val_loss": avg_val_loss,
                        "epoch": epoch,
                    },
                    "./checkpoint.pth",
                )
                wandb.save("checkpoint.pth")
            best_val_loss = avg_val_loss

        # End early in some circumstances
        # end = False
        # for name, kwargs in config.get("end_early", {}).items():
        #     end, message = ENDERS[name](
        #         losses["val"], epoch, config["epochs"], **kwargs
        #     )
        #     if end is True:
        #         print("*" * 80)
        #         print(f"ENDING EARLY\n{message}")
        #         print("*" * 80)
        #         break
        # if end is True:
        #     break

    if run is not None:
        run.finish()

    return avg_val_loss
