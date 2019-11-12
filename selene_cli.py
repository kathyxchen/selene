"""
Description:
    This script builds the model and trains it using user-specified input data.

Output:
    Saves model to a user-specified output file.

Usage:
    selene_cli.py <config-yml> [--lr=<lr>] [--rank=<rank>] [--world_size=<world_size>] [--gpu_id=<gpu_id>]
    selene_cli.py -h | --help

Options:
    -h --help                 Show this screen.

    <config-yml>              Model-specific parameters
    --lr=<lr>                 If training, the optimizer's learning rate
                              [default: None]
    --rank=<rank>             Rank of node for distributed training
                              [default: None]
    --world_size=<world_size> World size for distributed training
                              [default: None]
    --gpu_id=<gpu_id>         GPU id for distributed training
                              [default: None]
"""
from docopt import docopt
import torch
import torch.multiprocessing

from selene_sdk.utils import parse_configs_and_run
from selene_sdk import __version__


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version=__version__)

    torch.multiprocessing.set_start_method("spawn", force=True)
    if arguments["--lr"]:
        print("lr: {0}".format(arguments["--lr"]))
    if arguments["--rank"]:
        print("rank: {0}".format(arguments["--rank"]))
    if arguments["--world_size"]:
        print("world_size: {0}".format(arguments["--world_size"]))
    if arguments["--gpu_id"]:
        print("gpu_id: {0}".format(arguments["--gpu_id"]))

    print(arguments)
    parse_configs_and_run(arguments["<config-yml>"], lr=arguments["--lr"],
                          rank=arguments["--rank"], 
                          world_size=arguments["--world_size"],
                          gpu_id=arguments["--gpu_id"])
        
    
