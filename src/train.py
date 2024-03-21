import datetime
import gc
import torch
from tqdm import tqdm
import wandb

from scheduler import get_scheduler


def get_tools(loader, model, config):

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["wd"]
    )

    scheduler = get_scheduler(config, optimizer, loader)

    return (optimizer, scheduler)


def train_step(
    loader,
    model,
    optimizer,
    config,
    device,
    scheduler=None,
    log_loss=False,
    # log_training_images=False,
    # train_augmentation_path=None,
):

    model.train()
    batch_bar = tqdm(
        total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Train"
    )
    train_loss = 0
    # result = {"impaths": [], "losses": [], "outputs": []}

    # Reset this place to save evaluation outputs
    model.outputs["train"] = []

    for i, (x, y) in enumerate(loader):

        # if log_training_images:
        #     debug_impaths = save_debug_images(
        #         paths,
        #         Path("/tmp/"),
        #         from_torch=x,
        #         prefix=f"imgvis_{Path(train_augmentation_path).stem}_",
        #     )
        #     print(f"Saved debug images: {debug_impaths}")

        # Zero gradients (necessary to call explicitly in case you have split
        # training up across multiple devices)
        optimizer.zero_grad()

        x = x.to(device)
        y = y.to(device)
        loss = model.shared_step(x, y, "train")

        # # Do some bookkeeping, save these for later use
        # result["impaths"].extend(paths)
        # result["outputs"].extend([float(o) for o in out.detach().cpu()])

        loss.backward()
        optimizer.step()

        train_loss += float(loss.detach().cpu())
        batch_bar.set_postfix(
            loss=f"{train_loss/(i+1):.4f}", lr=f"{optimizer.param_groups[0]['lr']}"
        )
        batch_bar.update()

        # if log_loss and config["wandb"]:
        #     if i % config["train_report_iter"] == 0:
        #         wandb.log({"batch-loss": loss.item()})
        #         wandb.log({"batch-lr": float(scheduler.get_last_lr()[0])})
        if scheduler is not None:
            scheduler.step()

        del x, y, loss
        torch.cuda.empty_cache()

    batch_bar.close()

    # Return the average loss over all batches
    train_loss /= len(loader)

    print(
        f"{str(datetime.datetime.now())}"
        f"    Avg Train Loss: {train_loss:.4f}"
        f"    LR: {float(optimizer.param_groups[0]['lr']):.1E}"
    )

    return train_loss


def evaluate(loader, model, device):

    model.eval()

    val_loss = 0
    batch_bar = tqdm(
        total=len(loader), dynamic_ncols=True, leave=False, position=0, desc="Val"
    )

    # result = {"impaths": [], "losses": [], "outputs": [], "vectors": []}

    # Reset this place to save evaluation outputs
    model.outputs["val"] = []

    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.inference_mode():
            # # Sample the embeddings the first batch
            # if i == 0:
            #     modeldict = model(x, w_vec_embedding=True, w_im_embedding=True)
            #     embeddings = modeldict["embeddings"]
            # else:
            #     modeldict = model(x, w_vec_embedding=True)
            loss = model.shared_step(x, y, "val")

        val_loss += float(loss.detach().cpu())
        batch_bar.set_postfix(avg_loss=f"{val_loss/(i+1):.4f}")
        batch_bar.update()

        # # Do some bookkeeping, save these for later use
        # result["impaths"].extend(paths)
        # result["outputs"].extend(
        #     [float(o) for o in modeldict["outputs"].detach().cpu()]
        # )
        # result["losses"].extend([float(pil) for pil in per_input_loss.detach().cpu()])
        # result["vectors"].extend(
        #     [v.tolist() for v in modeldict["vectors"].detach().cpu()]
        # )

        del x, y, loss
        torch.cuda.empty_cache()

    batch_bar.close()

    # Get the average val_loss across the epoch
    val_loss /= len(loader)

    return val_loss


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
        # "train_augmentation_path": config["train_augmentation_path"],
    }
    if config["scheduler"] in ["constant", "CosMulti", "LRTest", "OneCycleLR"]:
        step_kwargs["scheduler"] = scheduler
        if config["scheduler"] == "LRTest":
            step_kwargs["log_loss"] = True

    # if debug:
    #     sanity_check(train_loader, model, device)

    best_val_loss = 1e6

    # losses = {"train": [], "val": []}
    sampled_paths = {}

    for epoch in range(config["epochs"]):
        print("Epoch", epoch + 1)

        train_loss = train_step(**step_kwargs)
        if config["scheduler"] == "StepLR":
            scheduler.step()
        # losses["train"].append(train_loss)

        log_values = {
            "train_loss": train_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        if (
            epoch % config["eval_report_iter"] == 0
            or epoch == config["epochs"] - 1
            or config["scheduler"] == "ReduceLROnPlateau"
        ):
            val_loss = evaluate(val_loader, model, device)
            print(f"{str(datetime.datetime.now())}    Validation Loss: {val_loss:.4f}")
            log_values.update({"val_loss": val_loss})
            # losses["val"].append(val_loss)
            if config["scheduler"] == "ReduceLROnPlateau":
                scheduler.step(val_loss)

        if config["wandb"]:
            wandb.log(log_values)

        if val_loss < best_val_loss:
            print("NOT saving model")
            # torch.save(
            #     {
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #         "val_loss": val_loss,
            #         "epoch": epoch,
            #     },
            #     "./checkpoint.pth",
            # )
            # if config["wandb"]:
            #     wandb.save("checkpoint.pth")
            best_val_loss = val_loss

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

    return val_loss
