# dgm-medfusion
Implementation of Medfusion for the DGM Image Statistics Challenge


## Requirements
* Docker
* Nvidia Drivers
* Nvidia Docker for GPU-Accelerated Containers

## Running Sampling

1. Run image and open console
    ```sh
    sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/:/workspace/project --rm nvcr.io/nvidia/pytorch:22.07-py3
    ```

2. Install requirements
    ```sh
    cd projects
    pip install -r requirements.txt
    ```

3. Run `sample_dataset` script
    ```sh
    python sample_dataset.py
    ```
