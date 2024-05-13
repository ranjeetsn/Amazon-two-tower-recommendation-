# Amazon two tower recommendation

- In this project we create a recommendation system using two-tower architecture

## Structure
- An insight to the structure of the source code:
```
🌳 project
├── README.md
├── services
│   ├── fastapi
│   │   ├── Dockerfile
│   │   ├── app.py
│   │   └── requirements.txt
├── test
│   └── test_example.py
```

## Demo


https://github.com/msds603-startup8/RehearsAI-server/assets/137837017/b0e32a40-1a4b-47dd-b729-7102827b0179



## Quick Start
### Local Environment Setup
1) Set Up OpenAI API Key:
   - Generate and export your OpenAI API key to enable API calls necessary for the application: ```export OPENAI_API_KEY=your_api_key_here```
2) Prepare Conda Environments:
   - Create two separate Conda environments to manage dependencies for Streamlit and FastAPI components:
   - Streamlit Environment: For running the Streamlit application.
   - FastAPI Environment: For handling API requests using FastAPI.
3) Install Required Packages:
   - Install the necessary Python packages in each environment:
```console
# Streamlit requirements
(streamlit) pip install -r ./services/streamlit/requirements.txt

# FastAPI requirements
(fastapi) pip install -r ./services/fastapi/requirements.txt
```
4) Run Servers
   - Launch both the Streamlit and FastAPI servers:
```console
# Streamlit server
(streamlit) cd services/streamlit; streamlit run app.py

# FastAPI server
(fastapi) uvicorn services.fastapi.app:app --port 8000 --reload
```

### Docker Container Setup
To deploy the application using Docker, follow these instructions:
1) Install Docker:
   - Install Docker if it is not already installed on your system.
2) Build Docker Images:
   - Create Docker images for both the Streamlit and FastAPI components:
```
# Build Streamlit image
docker build --no-cache -t streamlit -f ./services/streamlit/Dockerfile ./services/streamlit

# Build FastAPI image
docker build --no-cache -t fastapi -f ./services/fastapi/Dockerfile ./services/fastapi
```
3) Run the Servers:
- Start both servers using Docker:
```
# Run FastAPI server
docker run -e "OPENAI_API_KEY={your_openai_api_key}" -p 8000:8000 fastapi

# Run Streamlit server
docker run -e "OPENAI_API_KEY={your_openai_api_key}" -p 8080:8080 streamlit -- --host host.docker.internal
```
- Access the demo at http://localhost:8080.

### Follow these steps to run the demo and application locally:
1) Set-up OPENAI_API_KEY:
    - Creating and exporting your own OpenAI API key will allow you to make the necessary API calls to run our application. If you don't have an OpenAI key, visit the [OpenAI website.](https://platform.openai.com/signup)
```bash
export OPENAI_API_KEY=...
```

2) Prepare two conda environments:
   - The Streamlit environment will allow you to run the application in your local environment, and the the FastAPI environment will allow your machine to make the necessary calls to access the OpenAI APIs to run the model.
     - Streamlit
     - FastAPI
  - In order to run the concurrent environments, you will need two separate environments to allow the server to interact with the client side of the application.

  - Install required packages and libraries in respective conda environments:
```bash
(streamlit) pip install -r ./services/streamlit/requirements.txt
(fastapi) pip install -r ./services/fastapi/requirements.txt
```

  - Run the Streamlit server & FastAPI server
```bash
(streamlit) streamlit run ./services/streamlit/app.py
(fastapi) uvicorn services.fastapi.app:app --port 8000
```

## Quick Start (In Docker Container)
- In order to run the application, it is recommended that you Dockerize the Streamlit application and the FastAPI source code to allow the services to run concurrently and serve multiple users at the same time. 

- Install docker
https://docs.docker.com/engine/install/

- Set-up docker images:
```bash
docker build --no-cache \
    -t streamlit \
    -f ./services/streamlit/Dockerfile \
    ./services/streamlit

docker build --no-cache \
    -t fastapi \
    -f ./services/fastapi/Dockerfile \
    ./services/fastapi
```

- Run the web server and the client application:
```bash
# Run fastapi server
docker run -e "OPENAI_API_KEY={your_openai_api_key}" -p 8000:8000 fastapi

# Run streamlit server
docker run -e "OPENAI_API_KEY={your_openai_api_key}" -p 8080:8080 streamlit -- --host host.docker.internal
```

- Now you can see the demo at http://0.0.0.0:8080
