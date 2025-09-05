import math
import os

# set the current working directory as the project root directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from prettytable import PrettyTable
from torch.utils.data import Subset

from data.data import (
    DatasetOutputFormat,
    MultiEnvironmentDataset,
    TransformsGenerator,
)
from models import construct_model
from training import Trainer
from utils.utils import debug


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Number of params in total: ", pytorch_total_params)
    return total_params


def run(args):
    dataset_folder = f"{args.train.dataset_root_dpath}/{args.train.dataset_name}"
    # cache_dpath = (
    #     args.train.wandb_dpath if args.train.wandb_dpath != "./wandb" else "./cache"
    # )

    model = construct_model(args)
    num_frames = args.train.num_frames
    n_workers = args.train.n_data_workers
    n_total_samples = args.train.n_total_samples
    n_envs = args.train.n_envs

    # If a warm-start model path is provided, ensure it exists and load it
    if hasattr(args, "model_fpath") and args.model_fpath:
        if not os.path.exists(args.model_fpath):
            raise FileNotFoundError(
                f"Training warm-start checkpoint not found at '{args.model_fpath}'."
            )
        model_state_dict = torch.load(args.model_fpath, map_location="cpu")
        model.load_state_dict(model_state_dict["model"])  # strict loading
        del model_state_dict

    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        cpu=False,
        log_with="wandb",
        kwargs_handlers=[kwargs],
        mixed_precision="bf16",
    )

    transforms = TransformsGenerator.get_final_transforms(model.image_size, None)

    def get_data_handlers(n_workers=8, n_envs=0, n_total_samples=10000):
        train_ds_mev = MultiEnvironmentDataset(
            dataset_folder,
            seq_length_input=num_frames - 1,
            seq_step=1,
            split_type="session",
            split="train",
            transform=transforms["train"],
            format=DatasetOutputFormat.IVG,
            enable_cache=False,
            # cache_dpath=f"{cache_dir}/cache/{dataset_name}",
            n_workers=n_workers,
            n_envs=n_envs,
        )

        n_samples_valid = math.ceil(n_total_samples / train_ds_mev.n_datasets)

        valid_ds_mev = MultiEnvironmentDataset(
            dataset_folder,
            seq_length_input=num_frames - 1,
            seq_step=20,
            split_type="session",
            split="validation",
            transform=transforms["train"],
            format=DatasetOutputFormat.IVG,
            enable_cache=False,
            # cache_dpath=f"{cache_dir}/cache/{args.dataset}",
            n_workers=n_workers,
            n_envs=n_envs,
            n_samples=n_samples_valid,
        )
        valid_ds_mev = Subset(valid_ds_mev, range(n_total_samples))

        return train_ds_mev, valid_ds_mev

    # check if it is the main thread
    if accelerator.is_main_process:
        train_ds, valid_ds = get_data_handlers(
            n_workers=n_workers, n_total_samples=n_total_samples, n_envs=n_envs
        )

    accelerator.wait_for_everyone()

    if not accelerator.is_main_process:
        train_ds, valid_ds = get_data_handlers(
            n_workers=n_workers, n_total_samples=n_total_samples, n_envs=n_envs
        )

    save_dpath = f"{args.train.save_root_dpath}/{args.model}/{args.train.wandb_name}"

    trainer = Trainer(
        model=model,
        batch_size=args.train.batch_size,
        dataset=(train_ds, valid_ds),
        accelerator=accelerator,
        num_frames=args.train.num_frames,
        sample_num_frames=args.train.sample_num_frames,
        wandb_mode=args.train.wandb_mode,
        wandb_project=args.train.wandb_project,
        wandb_name=args.train.wandb_name,
        grad_accum_every=args.train.grad_accum,
        num_train_steps=args.train.num_train_steps,
        lr=args.optimizer.learning_rate,
        wd=args.optimizer.weight_decay,
        linear_warmup_start_factor=args.optimizer.linear_warmup_start_factor,
        linear_warmup_total_iters=args.optimizer.linear_warmup_total_iters,
        cosine_annealing_T_max=args.optimizer.cosine_annealing_t_max,
        cosine_annealing_eta_min=args.optimizer.cosine_annealing_min_lr,
        max_grad_norm=args.optimizer.max_grad_norm,
        save_dpath=save_dpath,
        save_model_every=args.train.save_model_every,
        wandb_dpath=args.train.wandb_dpath,
        max_valid_size=args.train.max_valid_size,
        validate_every=args.train.validate_every,
        train_config=args,
    )

    torch.cuda.empty_cache()

    if accelerator.is_main_process:
        debug("config: ", args)
        print("GenieRedux training is starting...\n")
        print("Dataset : ", dataset_folder)
        count_parameters(model)

    if args.train.resume_ckpt != "no":
        trainer.load(args.train.resume_ckpt)
        print("Model loaded from file:", args.train.resume_ckpt)
        print("Starting training from step:", trainer.step)

    model.train()
    trainer.train()
