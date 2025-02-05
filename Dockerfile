# Use an official Nvidia CUDA runtime as a parent image
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

# Avoid interactive prompts - auto-select defaults for any prompts 
ARG DEBIAN_FRONTEND=noninteractive

# Set timezone for tzdata package as it's a dependency for some packages
ENV TZ=America/Los_Angeles

# Set the working directory in the container
WORKDIR /app

# Install Python & PIP and git & wget to clone model repo and download model checkpoint
RUN apt-get update && apt-get install -y python3.10 python3-pip git wget vim unzip

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip install torch torchvision xformers
# Install PyTorch Nightly Build for CUDA 12.4, dependencies for Flash Attention 2 and initial dependencies for Kosmos-2.5
RUN pip install --pre torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 && \
	pip install -v wheel==0.37.1 ninja==1.11.1 packaging==24.1 numpy==1.22 psutil==6.0.0 && \
	pip install -v tiktoken tqdm "omegaconf<=2.1.0" boto3 iopath "fairscale==0.4" "scipy==1.10" triton flask

RUN pip install --verbose prebuilt_wheels/flash_attn-2.7.4.post1-cp310-cp310-linux_x86_64.whl --no-build-isolation

# Install remaining dependencies for Kosmos-2.5 from custom repos
RUN pip install -v git+https://github.com/Dod-o/kosmos2.5_tools.git@fairseq && \
	pip install -v git+https://github.com/Dod-o/kosmos2.5_tools.git@infinibatch && \
	pip install -v git+https://github.com/Dod-o/kosmos2.5_tools.git@torchscale && \
	pip install -v git+https://github.com/Dod-o/kosmos2.5_tools.git@transformers

# Clone model checkpoint
RUN wget -P /app/kosmos-2_5 https://huggingface.co/microsoft/kosmos-2.5/resolve/main/ckpt.pt

# Create image upload directory, no error if already exists
RUN mkdir -p /tmp

# Make port 25000 available to the world outside this container
EXPOSE 25000

# Set the Kosmos directory as the base application directory
WORKDIR /app/kosmos-2_5

# # Run application
# CMD ["python3", "kosmos_api.py"]

WORKDIR /app
CMD ["/bin/bash"]