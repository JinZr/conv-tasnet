import logging
import os

import torch
from torch import distributed as dist


def handle_scp(scp_path):
    """
    Read scp file script
    input:
          scp_path: .scp file's file path
    output:
          scp_dict: {'key':'wave file path'}
    """
    scp_dict = dict()
    line = 0
    lines = open(scp_path, "r").readlines()
    for l in lines:
        scp_parts = l.strip().split()
        line += 1
        if len(scp_parts) != 2:
            raise RuntimeError(
                "For {}, format error in line[{:d}]: {}".format(
                    scp_path, line, scp_parts
                )
            )
        if len(scp_parts) == 2:
            key, value = scp_parts
        if key in scp_dict:
            raise ValueError("Duplicated key '{0}' exists in {1}".format(key, scp_path))

        scp_dict[key] = value

    return scp_dict


def get_logger(
    name,
    format_str="%(asctime)s [%(pathname)s:%(lineno)s - %(levelname)s ] %(message)s",
    date_format="%Y-%m-%d %H:%M:%S",
    file=False,
):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # file or console
    handler = logging.StreamHandler() if not file else logging.FileHandler(name)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def setup_dist(
    rank, world_size, master_port=None, use_ddp_launch=False, master_addr=None
):
    """
    rank and world_size are used only if use_ddp_launch is False.
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = (
            "localhost" if master_addr is None else str(master_addr)
        )

    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12354" if master_port is None else str(master_port)

    if use_ddp_launch is False:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        dist.init_process_group("nccl")


def cleanup_dist():
    dist.destroy_process_group()


if __name__ == "__main__":
    print(len(handle_scp("/home/likai/data1/create_scp/cv_s2.scp")))
