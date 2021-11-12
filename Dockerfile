FROM python:3.8-slim

# Setup user
RUN useradd -m kapet
USER kapet
WORKDIR /home/kapet

# Install requirements.txt
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
