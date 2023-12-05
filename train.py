import argparse

import torch.multiprocessing as mp

from conv_tasnet import ConvTasNet
from dataloaders import make_dataloader
from options.option import parse
from trainer import Trainer
from utils import get_logger


def main():
    # Reading option
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, help="Path to option YAML file.")
    args = parser.parse_args()

    opt = parse(args.opt, is_tain=True)
    logger = get_logger(__name__)

    logger.info("Building the model of Conv-TasNet")
    net = ConvTasNet(**opt["net_conf"])

    logger.info("Building the trainer of Conv-TasNet")
    gpuid = tuple(opt["gpu_ids"])
    world_size = opt["world_size"]
    trainer = Trainer(
        net,
        **opt["train"],
        resume=opt["resume"],
        gpuid=gpuid,
        optimizer_kwargs=opt["optimizer_kwargs"],
        world_size=world_size,
    )

    logger.info("Making the train and test data loader")
    train_loader = make_dataloader(
        is_train=True,
        data_kwargs=opt["datasets"]["train"],
        num_workers=opt["datasets"]["num_workers"],
        chunk_size=opt["datasets"]["chunk_size"],
        batch_size=opt["datasets"]["batch_size"],
    )
    val_loader = make_dataloader(
        is_train=False,
        data_kwargs=opt["datasets"]["val"],
        num_workers=opt["datasets"]["num_workers"],
        chunk_size=opt["datasets"]["chunk_size"],
        batch_size=opt["datasets"]["batch_size"],
    )
    logger.info(
        "Train data loader: {}, Test data loader: {}".format(train_loader, val_loader)
    )

    assert world_size >= 1
    if world_size > 1:
        mp.spawn(
            trainer.run,
            args=(world_size, train_loader, val_loader),
            nprocs=world_size,
            join=True,
        )
    else:
        trainer.run(
            rank=0,
            world_size=world_size,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
        )


if __name__ == "__main__":
    main()
