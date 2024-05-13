# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

COPY ./requirements.txt /app/requirements.txt

# Install any needed dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port number the Streamlit server listens on
EXPOSE 8501

# Run main.py when the container launches
CMD ["streamlit", "run", "main.py"]
