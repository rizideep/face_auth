# Use an official Python runtime as the base image
FROM python:3.9-slim

 
# Upgrade pip and setuptools
RUN pip install --upgrade pip setuptools

# Install correct versions of numpy and opencv
RUN pip install numpy==1.23.5 

# Install dependencies
RUN pip install fastapi python-multipart  # Add other dependencies as needed

# Set the working directory inside the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app


# Install system dependencies required for building packages
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*



# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


# Expose the required port (if running a Flask or FastAPI app)
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
