import warnings

import hydra
import torch
import torch.multiprocessing as mp
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

from src.datasets.data_utils import get_dataloaders
from src.trainer import Trainer
from src.utils.ddp import setup
from src.utils.init_utils import set_random_seed, setup_saving_and_logging

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="baseline")
def main(config):
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size, config), nprocs=world_size, join=True)


def train(rank, world_size, config):
    """
    Main script for training. Instantiates the models, optimizers, schedulers,
    metrics, logger, writer, and dataloaders. Runs Trainer to train and
    evaluate the model. Also supports DDP training.

    Args:
        rank (int): rank of proccess for distributed training.
        world_size (int): total number of availiable GPUs.
        config (DictConfig): hydra experiment config.
    """
    if rank == 0:
        print(f"World size is {world_size}")

    setup(rank, world_size)
    set_random_seed(config.trainer.seed + rank)

    OmegaConf.register_new_resolver("devide", lambda x, y: x // y)
    project_config = OmegaConf.to_container(config)
    logger = setup_saving_and_logging(config)
    writer = instantiate(config.writer, logger, project_config)

    torch.backends.cudnn.benchmark = True
    device = torch.device(f"cuda:{rank}")

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, datasamplers, batch_transforms = get_dataloaders(
        config, world_size, rank
    )

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    ddp_model = DDP(model, device_ids=[rank])
    if rank == 0:
        logger.info(ddp_model)

    # get function handles of loss and metrics
    loss_function = instantiate(config.loss_function).to(device)
    metrics = instantiate(config.metrics)

    # build optimizer, learning rate scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = instantiate(config.optimizer, params=trainable_params)
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # epoch_len = number of iterations for iteration-based training
    # epoch_len = None or len(dataloader) for epoch-based training
    epoch_len = config.trainer.get("epoch_len")

    trainer = Trainer(
        model=ddp_model,
        criterion=loss_function,
        metrics=metrics,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        config=config,
        device=device,
        dataloaders=dataloaders,
        datasamplers=datasamplers,
        epoch_len=epoch_len,
        logger=logger,
        writer=writer,
        rank=rank,
        batch_transforms=batch_transforms,
        skip_oom=config.trainer.get("skip_oom", True),
    )

    trainer.train()


if __name__ == "__main__":
    main()
