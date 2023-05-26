#!/bin/bash

# Sample Dataset
# João Pedro Vasconcelos - jpvteixeira99@gmail.com
# First Version: 2023-05-22

# colors
NC='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'

AUTOENCODER_CHECKPOINT="$1"
PIPELINE_CHECKPOINT="$2"
OUTPUT_PATH="$3"
STEPS="$4"
SEED="$5"
BATCH_SIZE="$6"
DATASET_SIZE="$7"
NUM_WORKERS="$8"

# Check if all arguments were passed
if [ "$#" -ne 8 ]; then
    echo -e "${RED}Warning: Illegal number of parameters. Please pass all arguments.${NC}"
    echo -e "Usage: $0 <autoencoder_checkpoint> <pipeline_checkpoint> <output_path> <steps> <seed> <batch_size> <dataset_size> <num_workers>"
    exit 1
fi


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