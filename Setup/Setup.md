# Libraries used to create the dataset

The dataset has been created independently from the main workflow line using separate virtual environment in Anaconda utilising Python 3.9


```
conda create --name tiles gdal python==3.9
pip install geotile
pip install albumentations
pip install jupyter
pip install torch
```
gdal 3.6.2
geotile 1.1.0
albumentations 2.0.8
jupyter 1.1.1
torch  2.8.0

# Docker file

The Docker file is suitable to train the model, validate and predict using GPU in own local machine. It utilizes the Jupyter notebook library run from Docker container

It was tested on:
- Ubuntu 24.04.1 LTS (GNU/Linux 5.15.146.1-microsoft-standard-WSL2 x86_64) installed on WSL on Windows
- Ubuntu 24.04.1 installed on Proxmox as virtual machine

## Instructions
1. Clone the repository
```
git clone https://github.com/nazarb/2025_levees_DL.
```
2. Change directory to Dockerfile

```
cd 2025_levees_DL
cd Setup\unetr_docker

```
3. Build the Docker using the following command:
```
sudo docker build -t unetr .
```
4. Run the Docker container (change the {user} and entire folder location for your machine)
```
sudo docker run --gpus all -it --name unetr -v /home/{user}/Workspace:/Workspace -p 8888:8888 -p 9453:9453 --shm-size=32g unetr
```

Other comments
- Use sudo if necessary
- Adjust port numbers and location

```
sudo docker start unetr
sudo docker attach unetr
```

### Libraries used in the Docker container

The Docker file, which is accesible [here](https://github.com/nazarb/2025_levees_DL/blob/6e94ac25a49c68b2f58430cddac4cc31ffecbcb3/Docker/unetr_docker/Dockerfile) is created using the following code:


```
# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Install dependencies
RUN apt-get update && \
    apt-get install -y python3-pip && \
    pip3 install --upgrade pip

# Install Jupyter Lab
RUN pip install jupyterlab && \
    pip install notebook

# Install compatible Python packages
RUN pip install torch==2.0.1 \
                torchvision==0.15.2 \
                monai==1.2.0 \
                lightning==2.0.7 \
                einops==0.6.1 \
                numpy==1.25.2 \
                matplotlib==3.7.2 \
                rasterio==1.3.8 \
                scikit-learn==1.3.0 \
                opendatasets==0.1.22 \
                tiffile

# Set the working directory - change into your directory
WORKDIR /Workspace 

# Expose the necessary ports
EXPOSE 8888 9453

# Command to run Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

```

