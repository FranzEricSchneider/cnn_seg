# Built to enable running on an EC2 instance that is almost already there:
# Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.1.0 (Ubuntu 20.04) 20240312
pip3 install opencv-python
pip3 install pytorch_lightning
pip3 install segmentation-models-pytorch
pip3 install torchseg
pip3 install wandb
echo "Run with python3 cnn_seg/main.py <directory>"