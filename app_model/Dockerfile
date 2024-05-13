FROM --platform=arm64 python:3.10.12

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for PyQt5
# RUN apt-get update && apt-get install -y \
#     qt5-qmake \
#     qt5-qmake-bin \
#     qtbase5-dev \
#     qttools5-dev-tools \
#     && rm -rf /var/lib/apt/lists/*

# Install pip tools for building Python packages with native extensions
# RUN pip install --upgrade pip setuptools wheel

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
