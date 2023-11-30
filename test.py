import os
import time
from typing import Tuple
import unittest
import faiss
import torch
import logging
import numpy as np
from matplotlib import pyplot as plt
from numpy import ndarray
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from tools import commons
import datasets_ws
import parser
from model import network
from collections import OrderedDict
from os.path import join
from datetime import datetime

from datasets_ws import shift_window_on_descriptor  # import shift_window_similar calculate func
from model.sync_batchnorm import convert_model
from tools.visual import display_inference


def test(args, eval_ds, model, test_method="hard_resize", pca=None, show_inference_results=None, save_path=None):
    """Compute features of the given dataset and compute the recalls."""

    assert test_method in ["hard_resize"], f"test_method can't be {test_method}"

    model = model.eval()
    eval_ds.test_method = test_method

    with torch.no_grad():
        logging.debug("Extracting database features for evaluation/testing")

        database_features = np.empty((eval_ds.database_num, args.split_nums * args.features_dim), dtype="float32")
        database_subset_ds = Subset(eval_ds, list(range(eval_ds.database_num)))
        database_dataloader = DataLoader(dataset=database_subset_ds, num_workers=args.num_workers,
                                         batch_size=args.infer_batch_size, pin_memory=(args.device == "cuda"))

        # Database inputs shape : B, N, C, resize_H, resize_W
        for inputs, indices in tqdm(database_dataloader, ncols=100, desc='Extracting database features'):
            B, C, H, W = inputs.shape
            inputs = torch.stack([datasets_ws.shift_window_on_img(one_pano, eval_ds.split_nums, eval_ds.window_stride,
                                                                  eval_ds.window_len) for one_pano in inputs])
            inputs = inputs.view(B * eval_ds.split_nums, C, eval_ds.resize[0], eval_ds.resize[1])

            features = model(inputs.to(args.device))
            # B*split_nums, feature_dim -> # B, split_nums*feature_dim
            features = torch.flatten(features.view(B, eval_ds.split_nums, -1), start_dim=1)
            features = features.cpu().numpy()
            if pca != None:
                features = pca.transform(features)

            database_features[indices.numpy(), :] = features

        logging.debug("Extracting queries features for evaluation/testing")

        queries_infer_batch_size = args.infer_batch_size

        queries_features = np.empty((eval_ds.queries_num, args.features_dim), dtype="float32")
        queries_subset_ds = Subset(eval_ds,
                                   list(range(eval_ds.database_num, eval_ds.database_num + eval_ds.queries_num)))
        queries_dataloader = DataLoader(dataset=queries_subset_ds, num_workers=args.num_workers,
                                        batch_size=queries_infer_batch_size, pin_memory=(args.device == "cuda"))
        
        # Query features shape: B, C, H, W
        for inputs, indices in tqdm(queries_dataloader, ncols=100, desc='Extracting queries features'):
            features = model(inputs.to(args.device))
            features = features.cpu().numpy()

            if pca != None:
                features = pca.transform(features)

            # NOTE!! minus database_num to begin from 0
            queries_features[indices.numpy() - eval_ds.database_num, :] = features

    # Sliding Window Matching Descriptor
    shift_window_start = time.time()
    predictions = []
    focus_patch_loc = []
    for one_query_feature in queries_features:
        predictions_per_query, focus_patch_loc_per_query = shift_window_on_descriptor(one_query_feature,
                                                                                      database_features,
                                                                                      args.features_dim,
                                                                                      args.reduce_factor,
                                                                                      max(args.recall_values))
        predictions.append(predictions_per_query)
        focus_patch_loc.append(focus_patch_loc_per_query)  # show results interface
    shift_window_end = time.time()
    print(f'Searching all query in pano databases uses time:{shift_window_end-shift_window_start:.3f}s')

    # Visualization of Inference Results
    if show_inference_results:
        os.makedirs(save_path, exist_ok=True)
        display_inference(eval_ds, predictions, save_path, focus_patch_loc)

    #### For each query, check if the predictions are correct
    check_start = time.time()
    positives_per_query = eval_ds.get_positives()
    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / eval_ds.queries_num * 100
    recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(args.recall_values, recalls)])
    check_end = time.time()
    print(f'Checking whether the predict is right uses time:{check_end-check_start:.3f}s')
    return recalls, recalls_str


def main():
    # Initial setup: parser
    args = parser.parse_arguments()

    # Set Logger
    start_time = datetime.now()
    args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + args.title)
    commons.setup_logging(args.save_dir)
    logging.info(f"The outputs are being saved in {args.save_dir}")

    # Initialize model
    model = network.GeoLocalizationNet(args)
    model = model.to(args.device)

    # Muti-GPU Setting
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # val_ds
    val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
    logging.debug(f"Val set: {val_ds}")

    # test_ds
    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    logging.debug(f"Test set: {test_ds}")

    # load model params and run
    best_model_state_dict = torch.load(join(args.resume, "best_model.pth"))["model_state_dict"]

    if not torch.cuda.device_count() >= 2:
        best_model_state_dict = OrderedDict({k.replace('module.', ''): v for (k, v) in best_model_state_dict.items()})

    model.load_state_dict(best_model_state_dict)
    logging.info('Load pretrained model correctly!')

    recalls, recalls_str = test(args, eval_ds=val_ds, model=model)
    logging.info(f"Recalls on [Val-set]:{val_ds}: {recalls_str}")

    recalls, recalls_str = test(args, eval_ds=test_ds, model=model, show_inference_results=None)
    logging.info(f"Recalls on [Test-set]:{test_ds}: {recalls_str}")


if __name__ == '__main__':
    main()
