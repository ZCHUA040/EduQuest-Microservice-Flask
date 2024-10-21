# EduQuest-Microservice-Flask

This is a microservice for the EduQuest application. The main purpose is to invoke an LLM to retrieve documents and generate questions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/xeroxis-xs/eduquest-microservice-flask.git
    cd eduquest-microservice-flask
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To use this microservice, you need to have the necessary environment variables set up and the Flask application running.

## Configuration

Create a `.env` file in the root directory of the project and add the following environment variables:
```env
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
AZURE_OPENAI_API_VERSION=your_azure_openai_api_version
AZURE_OPENAI_ENDPOINT=yopur_azure_openai_endpoint
AZURE_OPENAI_DEPLOYMENT_NAME=yopur_azure_openai_deployment_name
AZURE_OPENAI_TEMPERATURE=your_azure_openai_temperature
AZURE_STORAGE_CONNECTION_STRING=your_azure_storage_account_connection_string
AZURE_STORAGE_CONTAINER_NAME=your_azure_storage_account_container_name
```

## Running the Application

### Option 1 (Recommended): Using JetBrains PyCharm
1. Select the Run/Debug Configuration dropdown in the top right corner of the PyCharm window.

![img.png](img.png)
2. Select Flask from the dropdown menu.
3. Click on the green play button to run the application.

![img_1.png](img_1.png)

### Option 2: Using the Command Line
1. Activate the virtual environment:
    ```sh
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
2. Run the Flask application:
    ```sh
    flask run
    ```
