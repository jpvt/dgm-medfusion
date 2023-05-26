from pathlib import Path
from datetime import datetime

import torch 
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from medical_diffusion.data.datamodules import SimpleDataModule
from medical_diffusion.data.datasets import DGMDataset
from medical_diffusion.models.embedders.latent_embedders import VAE
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from tqdm import tqdm
import torchvision.transforms.functional as tF
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mmssim
import json

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def eval_lpips(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    calc_lpips = LPIPS().to(device)
    for real_batch in tqdm(dataloader):
        imgs_real_batch = real_batch[0].to(device)

    imgs_real_batch = tF.normalize(imgs_real_batch/255, 0.5, 0.5) # [0, 255] -> [-1, 1]
    with torch.no_grad():
        imgs_fake_batch = model(imgs_real_batch)[0].clamp(-1, 1) 

    # -------------- LPIP -------------------
    calc_lpips.update(imgs_real_batch, imgs_fake_batch) # expect input to be [-1, 1]

    return 1-calc_lpips.compute()

def eval_msssim(model, dataloader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mmssim_list = []
    for real_batch in tqdm(dataloader):
        imgs_real_batch = real_batch[0].to(device)

    imgs_real_batch = tF.normalize(imgs_real_batch/255, 0.5, 0.5) # [0, 255] -> [-1, 1]
    with torch.no_grad():
        imgs_fake_batch = model(imgs_real_batch)[0].clamp(-1, 1) 

     # -------------- MS-SSIM -------------------
    for img_real, img_fake in zip(imgs_real_batch, imgs_fake_batch):
        img_real, img_fake = (img_real+1)/2, (img_fake+1)/2  # [-1, 1] -> [0, 1]
        mmssim_list.append(mmssim(img_real[None], img_fake[None], normalize='relu')) 

    return torch.mean(mmssim_list), torch.std(mmssim_list)


def train_vae(emb_dim, datamodule):
    # ------------ Initialize Model ------------
    model = VAE(
        in_channels=1, 
        out_channels=1, 
        emb_channels=emb_dim,
        spatial_dims=2,
        hid_chs =    [ 64, 128, 256,  512], 
        kernel_sizes=[ 3,  3,   3,    3],
        strides =    [ 1,  2,   2,    2],
        deep_supervision=1,
        use_attention= 'none',
        loss = torch.nn.MSELoss,
        embedding_loss_weight=0
    )

    # -------------- Training Initialization ---------------
    to_monitor = "train/L1"  # "val/loss" 
    min_max = "min"
    save_and_sample_every = 200

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=3, # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=5,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator='gpu',
        devices=[0],
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every, 
        auto_lr_find=False,
        max_epochs=10,
        num_sanity_val_steps=2,
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=datamodule)

    # ---------------- Execute Testing ----------------
    results = {}
    try:
        print("Calculating val loss")
        results["val_loss"] = trainer.test(model, dataloaders=datamodule.val_dataloader())
    except:
        results["val_loss"] = "Error"
        print("Error calculating val loss")

    try:
        print("Calculating test loss")
        results["test_loss"] = trainer.test(model, dataloaders=datamodule.test_dataloader())
    except:
        results["test_loss"] = "Error"
        print("Error calculating test loss")

    try:
        print("Calculating LPIPS")
        results["test_lpips"] = eval_lpips(model, datamodule.val_dataloader())
    except:
        results["test_lpips"] = "Error"
        print("Error calculating LPIPS")

    try:
        print("Calculating MS-SSIM")
        results["test_msssim"], results["test_msssim_std"] = eval_msssim(model, datamodule.val_dataloader())
    except:
        results["test_msssim"], results["test_msssim_std"] = "Error", "Error"
        print("Error calculating MS-SSIM")

    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)

    return model, results


if __name__ == "__main__":

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / str(current_time)
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None


    # ------------ Load Data ----------------
    ds = DGMDataset( #  512x512
        crawler_ext='png',
        augment_horizontal_flip=False,
        augment_vertical_flip=False,
        path_root=r'D:\Users\UFPB\jpvt\Diffusion Phantoms\DGMChallenge\challenge_data'
    )
   
    train_val_size = int(0.7 * len(ds))
    test_size = len(ds) - train_val_size
    train_size = int(0.8 * train_val_size)
    val_size = train_val_size - train_size
    print("Train Size, Val Size = ", train_size, val_size)
    print("Test Size = ", test_size)

    train_val_set, test_set = torch.utils.data.random_split(ds,[train_val_size, test_size])
    train_set, val_set = torch.utils.data.random_split(train_val_set,[train_size, val_size])

    dm = SimpleDataModule(
        ds_train = train_set,
        ds_val=val_set,
        ds_test=test_set,
        batch_size=10, 
        pin_memory=True
    ) 
    
    model_dict = {}
    for emb_dim in [4, 8, 16, 32]:
        print("Training model with emb_dim = ", emb_dim)
        model, results = train_vae(emb_dim, dm)
        model_dict[emb_dim] = {"results": results}

    # --------------- Save Results --------------------
    with open(path_run_dir / "results.json", 'w') as f:
        json.dump(model_dict, f, indent=4)

