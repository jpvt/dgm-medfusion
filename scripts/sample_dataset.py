from pathlib import Path
import torch
import math
from medical_diffusion.models.pipelines import DiffusionPipeline
from medical_diffusion.models.embedders.latent_embedders import VAE
from torchvision.utils import save_image
import time
from joblib import Parallel, delayed
from tqdm import tqdm

# ------------ Config ------------
AUTOENCODER_CHECKPOINT = 'model_checkpoints/vae.ckpt'
PIPELINE_CHECKPOINT = 'model_checkpoints/pipeline.ckpt'
OUTPUT_PATH = 'generated_datasets/'
STEPS = 300
SEED = 0
BATCH_SIZE = 100
DATASET_SIZE = 10000
NUM_WORKERS = 4

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def save_image_batch(image, path_out, counter):
    image = (image+1)/2  # Transform from [-1, 1] to [0, 1]
    image = image.clamp(0, 1)
    save_image(image, path_out/f'fake_{counter}_{time.time()}.png', nrow=int(math.sqrt(image.shape[0])), normalize=True, scale_each=True)
    counter += 1

# ------------ Load Model ------------
device = torch.device('cuda')
pipeline = DiffusionPipeline.load_from_checkpoint(
    PIPELINE_CHECKPOINT,
    latent_embedder=VAE,
    latent_embedder_checkpoint=AUTOENCODER_CHECKPOINT
)
pipeline.to(device)

if __name__ == "__main__":
    name=None
    label=None
    n_samples = DATASET_SIZE
    sample_batch = BATCH_SIZE
    cfg = 1

    path_out = Path(OUTPUT_PATH)
    path_out.mkdir(parents=True, exist_ok=True)

    # --------- Generate Samples  -------------------
    torch.manual_seed(SEED)
    counter = 0
    start_time_step = time.time()
    for chunk in chunks(list(range(n_samples)), sample_batch):
        start_chunk = time.time()
        condition = torch.tensor([label]*len(chunk), device=device) if label is not None else None 
        un_cond = torch.tensor([1-label]*len(chunk), device=device)  if label is not None else None # Might be None, or 1-condition or specific label 
        results = pipeline.sample(len(chunk), (8, 64, 64), guidance_scale=cfg, condition=condition, un_cond=un_cond, steps=STEPS)

        results = results.cpu()
        # --------- Save result ----------------
        Parallel(NUM_WORKERS)(
            delayed(save_image_batch)(
                image,
                path_out,
                counter
            ) for image in tqdm(results)
        )

        end_chunk = time.time()
        print(f"Chunk: {chunk} | Time: {end_chunk-start_chunk}")

    end_time_step = time.time()
    torch.cuda.empty_cache()
    print(f"Steps: {STEPS} | Total time: {end_time_step-start_time_step}")
    time.sleep(3)
