FROM --platform=arm64 python:3.10.12

# Set the working directory in the container
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    python3-dev

COPY . /app

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

RUN pip install -r requirements.txt

# Expose the port number the FastAPI server listens on
EXPOSE 80

# Run the application
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]
