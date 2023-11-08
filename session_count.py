# Assigns entropy values with a given model to the dataset in the order it appears.
import argparse
import bisect
import inspect
import os
import pickle
import sys
from collections import defaultdict
from functools import partial

import pandas as pd
import scipy.stats
import torch
import yaml
from matplotlib.axes import Axes
from torch.utils.data import DataLoader, SequentialSampler, BatchSampler, ConcatDataset
from tqdm import tqdm
from tabulate import tabulate
from matplotlib import pyplot as plt

from model.model import EHRAuditGPT2, EHRAuditRWKV, EHRAuditLlama
from model.modules import EHRAuditPretraining, EHRAuditDataModule, collate_fn, worker_fn
from model.data import timestamp_space_calculation
from model.vocab import EHRVocab, EHRAuditTokenizer
import numpy as np

# Fyi: this is a quick-and-dirty way of id'ing the columns, will need to be changed if the tabularization changes
METRIC_NAME_COL = 0
PAT_ID_COL = 1
ACCESS_TIME_COL = 2
USER_ID_COL = 3


if __name__ == "__main__":
    # Get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=int, default=None, help="Model to use for pretraining."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run with single thread.",
    )
    parser.add_argument(
        "--provider_type",
        type=str,
    )
    args = parser.parse_args()
    # Get the list of models from the config file
    config_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "config.yaml")
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    path_prefix = ""
    for prefix in config["path_prefix"]:
        if os.path.exists(prefix):
            path_prefix = prefix
            break

    if path_prefix == "":
        raise RuntimeError("No valid drive mounted.")

    # Load the test dataset
    vocab = EHRVocab(
        vocab_path=os.path.normpath(os.path.join(path_prefix, config["vocab_path"]))
    )

    dm = EHRAuditDataModule(
        yaml_config_path=config_path,
        vocab=vocab,
        batch_size=1,  # Just one sample at a time
        reset_cache=False,
        debug=args.debug,
        provider_type=args.provider_type,
    )
    dm.setup()

    all_datasets = dm.cdsets
    train_subset_indices = dm.train_dataset.indices
    tv_subset_indices = dm.val_dataset.indices + dm.test_dataset.indices
    comb_subset = torch.utils.data.Subset(all_datasets, tv_subset_indices)

    print(f"Counting chunks for {args.provider_type}")

    options = [
        ("train", train_subset_indices, dm.train_dataset),
        ("val/test", tv_subset_indices, comb_subset),
    ]

    for name, indices, dset in options:
        dl = torch.utils.data.DataLoader(
            dset,
            num_workers=0,
            batch_size=1,
            collate_fn=partial(collate_fn, n_positions=dm.n_positions),
            worker_init_fn=partial(worker_fn, seed=dm.seed),
            pin_memory=True,
            shuffle=False,
            batch_sampler=BatchSampler(SequentialSampler(indices), batch_size=1, drop_last=False),
        )

        # Iterate through the dataloader. Count the number of sessions, as well as the number of rows.
        session_count = 0
        row_count = 0
        row_len = len(vocab.field_ids) - 1
        pbar = tqdm(dl)
        for batch in pbar:
            input_ids, labels = batch
            session_count += 1
            # Find the eos index
            nonzeros = (labels.view(-1) == -100).nonzero(as_tuple=True)
            if len(nonzeros[0]) == 0:
                eos_index = len(labels.view(-1)) - 1
            else:
                eos_index = nonzeros[0][0].item() - 1

            batch_rows = eos_index // row_len
            row_count += batch_rows
            pbar.set_postfix({"rows": row_count, "session_count": session_count})
        print(f"Total rows for {name}: {row_count}")
        print(f"Total sessions for {name}: {session_count}")
