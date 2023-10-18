# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV WEAVIATE_URL "http://weaviate:8080"

# Create a mount point for a volume on the host
VOLUME /app/data

# Run app.py when the container launches
CMD ["uvicorn", "embedding_generator:app", "--host", "0.0.0.0", "--port", "80"]