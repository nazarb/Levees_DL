# Docker file

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
## Instructions
1. Create a folder inside your machine and change directory into it:
```
mkdir unetr_docker
cd unetr docker
nano Dockerfile
```
2. Copy the content of the Dockerfile in Nano (Linux) and save the file.
3. Build the Docker using the following command:
```
sudo docker build -t Dockerfile_yolo_sam .
```
5. Run the Docker container (change the {user} and entire folder location for your machine)
```
sudo docker run --gpus all -it --name unetr -v /home/{user}/Workspace:/Workspace -p 8888:8888 -p 9453:9453 --shm-size=32g unetr
```
6. Other comments
- Use sudo if necessary
- Adjust port numbers and locatiion

```
sudo docker start unetr
sudo docker
```
