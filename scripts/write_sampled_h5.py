"""
Description:
    This script uses one of Selene's online samplers to create a
    dataset from a tabix-indexed BED file of peaks and a genome.

Usage:
    write_sampled_h5.py <config-yml> <mode> <n-steps> <rseed> <packbits>
    write_sampled_h5.py -h | --help

Options:
    -h --help               Show this screen.

    <config-yml>            Sampler parameters
    <mode>                  Sampling mode. Must be one of
                            {"train", "validate", "test"}.
    <n-steps>               Number of steps to take (total number of data
                            samples generated will be batch_size * n_steps
    <rseed>                 The random seed to use during sampling.
    <packbits>              Whether to pack bits or not.
"""
from docopt import docopt
import os

import h5py
import numpy as np

from selene_sdk.utils import load_path, instantiate


if __name__ == "__main__":
    arguments = docopt(
        __doc__,
        version="1.0")
    print(arguments)
    configs = load_path(arguments["<config-yml>"], instantiate=False)
    configs["sampler"].bind(
        mode=arguments["<mode>"],
        seed=int(arguments["<rseed>"]),
        save_datasets=[])
    output_dir = configs["sampler"].keywords["output_dir"]
    data_sampler = instantiate(configs["sampler"])

    seq_len = configs["sampler"].keywords["sequence_length"]
    batch_size = configs["batch_size"]
    n_steps = int(arguments["<n-steps>"])
    packbits = arguments["<packbits>"] == 'True'

    with h5py.File(os.path.join(output_dir,
                                "{0}_seed={1}_N={2}.h5".format(
                                    arguments["<mode>"],
                                    arguments["<rseed>"],
                                    batch_size * n_steps)), "a") as fh:
        seqs = None
        tgts = None
        for i in range(n_steps):
            if i % 50 == 0:
                print("processing step {0} for {1} records".format(i, arguments["<mode>"]))
            sequences, targets = data_sampler.sample(batch_size=configs["batch_size"])
            sequences_length = sequences.shape[1]
            targets_length = targets.shape[1]
            if packbits:
                sequences = np.packbits(sequences > 0, axis=1)
                targets = np.packbits(targets > 0, axis=1)
            if seqs is None:
                fh.create_dataset("sequences_length", data=sequences_length)
                if packbits:
                    seqs = fh.create_dataset(
                        "sequences",
                        (configs["batch_size"] * n_steps,
                         sequences.shape[1],
                         sequences.shape[2]),
                        dtype='uint8')
                else:
                    seqs = fh.create_dataset(
                        "sequences",
                        (configs["batch_size"] * n_steps,
                         sequences.shape[1],
                         sequences.shape[2]),
                        dtype='float')
            if tgts is None:
                fh.create_dataset("targets_length", data=targets_length)
                tgts = fh.create_dataset(
                    "targets",
                    (configs["batch_size"] * n_steps, targets.shape[1]),
                    dtype='uint8')
            seqs[i*configs["batch_size"]:(i+1)*configs["batch_size"]] = sequences
            tgts[i*configs["batch_size"]:(i+1)*configs["batch_size"],:] = targets
