# Amazon two tower recommendation

- In this project we create a recommendation system using two-tower architecture

## Structure
- An insight to the structure of the source code:
```
🌳 project
.
├── README.md
├── Testing-Model.ipynb
├── Two-Tower-model-training.ipynb
├── app_model
│   ├── Dockerfile
│   ├── app
│   │   └── app.py
│   ├── model
│   │   ├── __init__.py
│   │   ├── config.py
│   │   └── model_handler.py
│   └── requirements.txt
├── best_model_weights.keras
├── folder_structure.txt
├── models
│   ├── full_model.pth
│   ├── product_dictionary.json
│   ├── product_model.tree
│   └── recommendation_model.h5
├── product_model.tree
├── recommendation_model.h5
└── streamlit
    ├── Dockerfile
    ├── main.py
    └── requirements.txt
```

## Demo

[Watch the video](https://github.com/ranjeetsn/Amazon-two-tower-recommendation-/assets/30920464/c2a2df78-cec4-422e-9ee8-c8b916d4da96)







## Quick Start
# Two-Tower Recommendation System

This project implements a two-tower recommendation system using FastAPI and Streamlit. It features a FastAPI server to handle backend operations and a Streamlit application for the frontend interface.

## Local Environment Setup

### Prerequisites
- Python 3.10.12
- Conda (Anaconda or Miniconda)

### Setup Steps

1. **Clone the Repository**
   git clone [<repository-url>]https://github.com/ranjeetsn/Amazon-two-tower-recommendation-.git
   cd Amazon-two-tower-recommendation-

2. **Create Conda Environments**
   - Create and activate separate Conda environments for Streamlit and FastAPI:
     conda create -n streamlit python=3.10.12
     conda activate streamlit
     pip install -r ./streamlit/requirements.txt

     conda create -n fastapi python=3.10.12
     conda activate fastapi
     pip install -r ./app_model/requirements.txt

3. **Run Servers**
   - Start the FastAPI server:
     (fastapi) uvicorn app_model.app.app:app --host 0.0.0.0 --port 8000
   
   - Start the Streamlit server:
     (streamlit) streamlit run streamlit/main.py

## Docker Container Setup

### Build and Run Docker Containers

1. **Install Docker**
   - Ensure Docker is installed on your system by following the instructions at Docker Installation Guide (https://docs.docker.com/engine/install/).

2. **Build Docker Images**
   docker build --no-cache -t streamlit -f ./streamlit/Dockerfile ./streamlit
   docker build --no-cache -t fastapi -f ./app_model/Dockerfile ./app_model

3. **Run the Docker Containers**
   docker run -p 8000:8000 fastapi
   docker run -p 8080:8080 streamlit -- --host host.docker.internal

### Access the Application

- Open your web browser and access the Streamlit interface at http://localhost:8080.

## Support

For any issues or questions, please open an issue on the GitHub repository or contact the development team.
