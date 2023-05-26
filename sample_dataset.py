import argparse
from pathlib import Path
import torch
import math
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.embedders.latent_embedders import VAE
from torchvision.utils import save_image
import time
from joblib import Parallel, delayed
from tqdm import tqdm

# ------------ ArgParse ------------
parser = argparse.ArgumentParser(conflict_handler='resolve')
parser.add_argument("--ae_ckpt",    type=str,   default='project/model_checkpoints/vae.ckpt', help="Autoencoder weights")
parser.add_argument("--pip_ckpt",    type=str,   default='project/model_checkpoints/pipeline.ckpt', help="Diffusion Pipeline weights")
parser.add_argument("--output_dir",    type=str,   default='project/generated_datasets/', help="Generated Datasets output directory")
parser.add_argument("--steps",    type=int,   default=300, help="Number of steps for sampling")
parser.add_argument("--seed",    type=int,   default=0, help="Random seed")
parser.add_argument("--batch_size",    type=int,   default=20, help="Batch size")
parser.add_argument("--dataset_size",    type=int,   default=10000, help="Number of samples to generate")
parser.add_argument("--num_workers",    type=int,   default=4, help="Number of workers for dataloader")
args = parser.parse_args()
# --------------------------------

# ------------ Config ------------
AUTOENCODER_CHECKPOINT = args.ae_ckpt
PIPELINE_CHECKPOINT = args.pip_ckpt
OUTPUT_PATH = args.output_dir
STEPS = args.steps
SEED = args.seed
BATCH_SIZE = args.batch_size
DATASET_SIZE = args.dataset_size
NUM_WORKERS = args.num_workers
LATENT_DIM_SHAPE = (8, 64, 64)
# --------------------------------

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def save_image_batch(image, path_out, counter):
    image = (image+1)/2  # Transform from [-1, 1] to [0, 1]
    image = image.clamp(0, 1)
    save_image(image, path_out/f'fake_{counter}_{time.time()}.png', nrow=int(math.sqrt(image.shape[0])), normalize=True, scale_each=True)
    counter += 1

def sample_chunk(pipeline, n_samples, guidance_scale=1,condition=None, un_cond=None, steps=100):
    results = pipeline.sample(n_samples, LATENT_DIM_SHAPE, guidance_scale=guidance_scale, condition=condition, un_cond=un_cond, steps=steps)
    results = results.cpu()
    return results

# ------------ Load Model ------------
device_gpu = torch.device('cuda')
pipeline = DiffusionPipeline.load_from_checkpoint(
    PIPELINE_CHECKPOINT,
    latent_embedder=VAE,
    latent_embedder_checkpoint=AUTOENCODER_CHECKPOINT
)

if __name__ == "__main__":
    n_samples = DATASET_SIZE
    sample_batch = BATCH_SIZE
    path_out = Path(OUTPUT_PATH)
    path_out.mkdir(parents=True, exist_ok=True)

    # --------- Generate Samples  -------------------
    torch.manual_seed(SEED)
    counter = 0
    start_time_step = time.time()
    for chunk in chunks(list(range(n_samples)), sample_batch):
        start_chunk = time.time()
        pipeline.to(device_gpu)
        results = sample_chunk(pipeline, len(chunk), steps=STEPS)
        pipeline.cpu()
        # --------- Save result ----------------
        Parallel(NUM_WORKERS)(
            delayed(save_image_batch)(
                image,
                path_out,
                counter
            ) for image in tqdm(results)
        )

        end_chunk = time.time()
        counter += 1
        print(f"Chunk: {counter} | Time: {end_chunk-start_chunk}")
        torch.cuda.empty_cache()
        time.sleep(3)

    end_time_step = time.time()
    print(f"Steps: {STEPS} | Total time: {end_time_step-start_time_step}")
