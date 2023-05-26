# dgm-medfusion
Implementation of Medfusion for the DGM Image Statistics Challenge


## Requirements
* Docker
* Nvidia Drivers
* Nvidia Docker for GPU-Accelerated Containers

## Running Sampling

1. Prepare the script to run
    ```sh
    chmod a+x sample.sh 
    ```

2. (Optional) Select [sampling parameters](#sampling-parameters)

3. Run the `sample.sh` script
    ```sh
    sudo ./sample.sh
    ```

## Sampling Parameters

**Default:**

```
AUTOENCODER_CHECKPOINT="project/model_checkpoints/vae.ckpt"
PIPELINE_CHECKPOINT="project/model_checkpoints/pipeline.ckpt"
OUTPUT_PATH="project/generated_datasets/"
STEPS="300"
SEED="0"
BATCH_SIZE="20"
DATASET_SIZE="10000"
NUM_WORKERS="4"
```

**Note:** You can change this parameters in the `sample.sh` script

### Parameters Description

* AUTOENCODER_CHECKPOINT: Path to the weights of the Variational Autoencoder used in the Diffusion Pipeline
* PIPELINE_CHECKPOINT: Path to the weights of the Diffusion Pipeline that generates the dataset
* OUTPUT_PATH: Path to the generated dataset directory
* STEPS: Steps in the sampling pipeline. 300 by default. If you want a faster sampling you can decrease to 100 or 200 in detriment of image quality. If you want higher quality in your images you can increase to 400, 500+ (This increases the sampling time significantly)
* SEED: Random seed used to generate the dataset
* BATCH_SIZE: Number of images generated each batch. 20 images by default. This batch size peaks ~15 GB VRAM usage. If you are training in a larger GPU, increase this value for faster sampling.
* DATASET_SIZE: Number of images generated in total. 10000 by default. In the default values, this usually takes about 8 hours.
* NUM_WORKERS: Number of CPUs used to write the images.
