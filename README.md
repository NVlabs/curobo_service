# Curobo Service

The Curobo Service is a [FastAPI](https://fastapi.tiangolo.com) http server which calls out to the curobo motion planning library.
You can hit the curobo service endpoints with whatever language you like.

## Install and run the Curobo Service in a virtual environment
* Navigate to the directory where you would like to clone the repos into and set it as an environment variable.
    ```
    cd <repos directory>
    export REPOS_DIR=$(pwd)
    ```
* Clone the curobo service repo into your repos directory.
    ```
    cd $REPOS_DIR
    git clone https://github.com/NVlabs/curobo_service.git
    cd curobo_service
    ```
* Set up a virtual environment and install the project into it.
    ```
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .
    ```
* Navigate to your repos directory and clone curobo into it.
    ```
    cd $REPOS_DIR
    git clone https://github.com/NVlabs/curobo.git
    cd curobo
    ```
* Install curobo and dependencies into the virtual environment.
    ```
    pip install torch==2.4.0
    pip install -e .
    ```
* Return to your `curobo_service` directory and run the tests.
    ```
    cd $REPOS_DIR/curobo_service
    make test
    ```
* Run the service:
    ```
    make run
    ```
You can verify that the service is running by visiting [http://127.0.0.1:10000/docs](http://127.0.0.1:10000/docs) in your browser.


## How the Curobo Service Works
Curobo service uses fastapi, which in turn uses Pydantic for data validation.
One file [src/nvidia/srl/curobo_service/main.py](src/nvidia/srl/curobo_service/main.py) contains all the endpoints for the service.

A different file [src/nvidia/srl/curobo_service/data_models.py](src/nvidia/srl/curobo_service/data_models.py) contains the pydantic data models (validated dataclasses) for the service.

The server is run with `uvicorn` in `src/nvidia/srl/curobo_service/main.py`.
See [uvicorn](https://www.uvicorn.org/) for more information.
