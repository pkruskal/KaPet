FROM python:3.8-slim

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
