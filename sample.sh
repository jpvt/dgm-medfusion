#!/bin/bash

# Sample Dataset
# João Pedro Vasconcelos - jpvteixeira99@gmail.com
# First Version: 2023-05-22

# colors
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'

# Arguments
AUTOENCODER_CHECKPOINT="project/model_checkpoints/vae.ckpt"
PIPELINE_CHECKPOINT="project/model_checkpoints/pipeline.ckpt"
OUTPUT_PATH="project/generated_datasets/"
STEPS="300"
SEED="0"
BATCH_SIZE="20"
DATASET_SIZE="10000"
NUM_WORKERS="4"

# Setup Environment
echo -e "${GREEN}Setting up environment...${NC}"

# building image
echo -e "⚙️  Building the custom docker image..."
sudo docker build . -t dgm_medfusion:latest
if [ "$?" -ne 0 ]; then
    echo -e "${RED}Error building image. Run $0 again.${NC} ❌"
    exit 1
else
    echo -e "${GREEN}Done!${NC} ✅"
fi

# Running container
sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it -v $(pwd)/:/workspace/project --rm dgm_medfusion:latest python ./project/sample_dataset.py --ae_ckpt $AUTOENCODER_CHECKPOINT --pip_ckpt $PIPELINE_CHECKPOINT --output_dir $OUTPUT_PATH --steps $STEPS --seed $SEED --batch_size $BATCH_SIZE --dataset_size $DATASET_SIZE --num_workers $NUM_WORKERS