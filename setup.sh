# RUN PYTORCH CONSOLE
# sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/pytorch:22.07-py3

# RUN PYTORCH CONSOLE WITH VOLUME
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/:/workspace/project --rm nvcr.io/nvidia/pytorch:22.07-py3