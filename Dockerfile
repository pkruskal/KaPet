FROM python:3.9-slim as kapet

# Install libraries
RUN apt-get update && apt-get install -y build-essential

# Setup user
RUN useradd -m kapet
USER kapet
WORKDIR /home/kapet

# Add user dir to path
ENV HOME=/home/kapet
ENV PATH="$HOME/.local/bin:$PATH"

# Install requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

####
# Adding GPU support with pytorch base
####
FROM anibali/pytorch:1.10.0-cuda11.3 as kapet_gpu

# Install libraries
USER root
RUN sudo apt-get update && apt-get install -y build-essential

# Setup user
RUN useradd -m kapet
USER kapet
WORKDIR /home/kapet

# Add user dir to path
ENV HOME=/home/kapet
ENV PATH="$HOME/.local/bin:$PATH"

# Install requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

