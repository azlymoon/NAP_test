# Use an official PyTorch image as a parent image, compatible with CUDA for GPU support
FROM mirror.gcr.io/pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Install Python 3.8 (if necessary) and any needed packages specified in requirements.txt
COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

# Upgrade numpy to meet matplotlib's requirements
RUN pip install --upgrade numpy

RUN pip install --upgrade torch torchvision

# Copy the entire project directory, respecting .dockerignore
COPY . /usr/src/app/

# Install ultralytics as a module if it's a Python package
RUN pip install ./ultralytics

# Define environment variable
ENV NAME Naturalistic-Adversarial-Patch

# Define command or entry point
CMD ["python", "ensemble.py"]
