# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /workspaces/deep-learning-experiment

# Copy only the requirements file first to leverage Docker caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the files
COPY . .

# Set environment variables
ARG WANDB_API_KEY
ENV WANDB_API_KEY=${WANDB_API_KEY}

# Initialize DVC repository
RUN dvc init

# Run dvc repro when the container launches
CMD ["dvc", "repro"]
