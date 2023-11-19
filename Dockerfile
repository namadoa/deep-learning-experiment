# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Copy the current directory contents into the container at /usr/src/app
COPY . /workspaces/modelling

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /workspaces/modelling/requirements.txt

# Set environment variables
ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

# Set the working directory in the container
WORKDIR /workspaces/modelling

# Initialize DVC repository
RUN dvc init

# Run dvc repro when the container launches
CMD ["dvc", "repro"]