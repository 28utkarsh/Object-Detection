# Object Detection using Single Shot MultiBox Detector (SSD)

This repository performs Object Detection using SSD algorithm. The code can be executed in the Docker environment. Therefore, your system MUST have docker and nvidia-docker [toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed.

### Environment Setup

Follow the instruction given below to set up the environment

```bash
# Clone this repository
# Set the following environment variable being in the same directory
export REPO_ROOT=`readlink -f Object-Detection/`

# Download the SSD model weights file from Weights File Download Section

# Pull the Docker Image
docker pull pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

# Start the container
docker run -it --gpus all --rm -v $REPO_ROOT:/object_detection -w /object_detection --name Object_Detection pytorch/pytorch:0.4.1-cuda9-cudnn7-devel

# Install dependencies
bash install_apt_deps.sh
pip install -r requirements.txt
```

### Run Inference

The inference command uses the following command line parameters:
1. **input_video** : Path to Input Video
2. **output_video** : Path to Output Video
3. **weights_file_path** : Path to SSD Pretrained Weigths file

```bash
python3 object_detection.py --input_video funny_dog.mp4 --output_video output.mp4 --weights_file_path ssd300_mAP_77.43_v2.pth
```

### Weights File Download
Please download the [weights file](https://drive.google.com/open?id=1FiN8I6tQykG1fjhrERtUd7FuXSFGXnLo) and keep it in the root of this repository.


### Reference

1. SSD Inference : [Deep Learning and Computer Learning Course](https://www.udemy.com/course/computer-vision-a-z/) at Udemy