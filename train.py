import os

import math
import torch
import logging
import numpy as np
from matplotlib import pyplot as plt
from torch.utils import data
from tqdm import tqdm
import torch.nn as nn
import multiprocessing
from os.path import join
from datetime import datetime
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

from tools import util
import test
import parser
from tools import commons
import datasets_ws
from model import network
from model.sync_batchnorm import convert_model
from model.functional import sare_ind, sare_joint
from tools.visual import display_mining
from tools.loss import shift_window_triple_loss


def train(args,
          model: nn.Module,
          train_set: datasets_ws.TripletsDataset,
          loss_fn: nn.TripletMarginLoss,
          optimizer,
          val_set: datasets_ws.BaseDataset,
          writer,
          start_epoch_num=0,
          best_r5=0,
          not_improved_num=0,
          show_mining_triplet_img=None,
          visual_mining_save_path=None,
          show_inference_results=None,
          visual_val_save_path=None
          ):
    """
    Trains a given model using triplet loss and performs validation.

    Args:
        args: A namespace or dictionary containing training parameters.
        model (nn.Module): The neural network model to train.
        train_set (datasets_ws.TripletsDataset): The training dataset containing triplets.
        loss_fn (nn.TripletMarginLoss): The loss function to optimize.
        optimizer: The optimization algorithm.
        val_set (datasets_ws.BaseDataset): The validation dataset.
        writer: A summary writer object for logging.
        start_epoch_num (int): The starting epoch number for training.
        best_r5 (float): The best recall@5 score obtained so far.
        not_improved_num (int): Counter for epochs without improvement.
        show_mining_triplet_img (callable, optional): Function to visualize mining results.
        visual_mining_save_path (str, optional): Path to save visual mining results.
        show_inference_results (callable, optional): Function to visualize inference results.
        visual_val_save_path (str, optional): Path to save validation visualization results.

    The training process includes mining hard triplets, calculating loss, and updating the model parameters.
    Validation is performed at the end of each epoch to monitor the recall metrics and early stopping is applied
    based on the recall@5 metric.
    """

    scaler = GradScaler()

    # Training loop
    for epoch_num in range(start_epoch_num, args.epochs_num):
        logging.info(f"Start training epoch: {epoch_num:02d}")

        epoch_start_time = datetime.now()
        epoch_losses = np.zeros((0, 1), dtype=np.float32)

        # How many loops should an epoch last
        loops_num = math.ceil(args.queries_per_epoch / args.cache_refresh_rate)
        for loop_num in range(loops_num):
            logging.debug(f"Cache: {loop_num} / {loops_num}")

            # Compute triplets to use in the triplet loss
            train_set.is_inference = True
            train_set.compute_triplets(args, model)
            train_set.is_inference = False

            # Visualizing mining results.
            if show_mining_triplet_img:
                os.makedirs(f'{visual_mining_save_path}' + f'mining_epoch{epoch_num}/', exist_ok=True)
                save_path = f'{visual_mining_save_path}' + f'mining_epoch{epoch_num}/loopNum{loop_num}_'
                triplets_global_indexes_array = train_set.triplets_global_indexes.numpy()
                # Obtaining the sub-window focused on by the model.
                pos_patch_loc, neg_patch_loc = train_setshift_window_on_img.pos_focus_patch, train_set.neg_focus_patch
                display_mining(train_set, triplets_global_indexes_array, save_path, pos_patch_loc, neg_patch_loc)

            triplets_dl = DataLoader(dataset=train_set, num_workers=args.num_workers,
                                     batch_size=args.train_batch_size,
                                     pin_memory=(args.device == "cuda"),
                                     drop_last=True)

            model = model.train()

            cache_losses = np.zeros((0, 1), dtype=np.float32)
            batch_nums = args.cache_refresh_rate / args.train_batch_size

            # images shape: (train_batch_size*12)*3*H*W
            for batch_idx, (query, pano_database) in enumerate(tqdm(triplets_dl, ncols=100, desc='Training')):
                # query shape:(B, 3, 224, 224)  ||  pano_database shape:(B, 11, 3, 224, 224*8)
                pano_database_4d = torch.flatten(pano_database, end_dim=1)  # B*11, 3, 224, 224*8

                pano_split_list = []
                for pano_full in pano_database_4d:
                    pano_split = datasets_ws.shift_window_on_img(pano_full, train_set.split_nums,
                                                                train_set.window_stride, train_set.window_len)

                    pano_split_list.append(pano_split)

                pano_database_4d = torch.flatten(torch.stack(pano_split_list), end_dim=1)

                optimizer.zero_grad()
                with autocast():
                    database_feature = model(pano_database_4d.to(args.device)).view(args.train_batch_size,
                                                                                    1 + args.negs_num_per_query, -1)
                    query_feature = model(query.to(args.device))

                    loss_triplet = 0
                    loss = 0
                    for idx in range(args.train_batch_size):
                        loss, min_index_in_row = shift_window_triple_loss(args, query_feature[idx], database_feature[idx], loss_fn)
                        loss_triplet += loss
                    del database_feature, query_feature

                    loss_triplet /= (args.train_batch_size * args.negs_num_per_query)

                scaler.scale(loss_triplet).backward()

                scaler.unscale_(optimizer)
                # clip_grad 1,3,5,7,10
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

                scaler.step(optimizer)
                scaler.update()

                # Keep track of all losses by appending them to epoch_losses
                batch_loss = loss_triplet.item()

                writer.add_scalar('Loss/Batch_Loss', batch_loss,
                                  epoch_num * loops_num * batch_nums + loop_num * batch_nums + batch_idx)

                cache_losses = np.append(cache_losses, batch_loss)
                del loss_triplet

            logging.debug(f"Epoch[{epoch_num:02d}]({loop_num}/{loops_num}): " +
                          f"latest batch triplet loss = {batch_loss:.4f}, " +
                          f"current cache triplet loss = {cache_losses.mean():.4f}")
            writer.add_scalar('Loss/Cache_Loss', cache_losses.mean().item(), epoch_num * loops_num + loop_num)

            # epoch_losses should update after calculating cache_loss
            epoch_losses = np.append(epoch_losses, cache_losses)

        logging.info(f"Finished epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                     f"average epoch triplet loss = {epoch_losses.mean():.4f}")
        writer.add_scalar('Loss/Epoch_Loss', epoch_losses.mean().item(), epoch_num)

        # Compute recalls on validation set
        recalls, recalls_str = test.test(args, val_set, model,
                                         show_inference_results=show_inference_results,
                                         save_path=visual_val_save_path)
        logging.info(f"Recalls on val set {val_set}: {recalls_str}")

        writer.add_scalar('Recall/@1', recalls[0], epoch_num)
        writer.add_scalar('Recall/@5', recalls[1], epoch_num)
        writer.add_scalar('Recall/@10', recalls[2], epoch_num)
        writer.add_scalar('Recall/@20', recalls[3], epoch_num)

        is_best = recalls[1] > best_r5

        # Save checkpoint, which contains all training parameters
        util.save_checkpoint(args, {"epoch_num": epoch_num, "model_state_dict": model.state_dict(),
                                    "optimizer_state_dict": optimizer.state_dict(), "recalls": recalls,
                                    "best_r5": best_r5,
                                    "not_improved_num": not_improved_num
                                    }, is_best, filename="last_model.pth")

        # If recall@5 did not improve for "many" epochs, stop training
        if is_best:
            logging.info(f"Improved: previous best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
            best_r5 = recalls[1]
            not_improved_num = 0
        else:
            not_improved_num += 1
            logging.info(
                f"Not improved: {not_improved_num} / {args.patience}: best R@5 = {best_r5:.1f}, current R@5 = {recalls[1]:.1f}")
            if not_improved_num >= args.patience:
                logging.info(f"Performance did not improve for {not_improved_num} epochs. Stop training.")
                break

    logging.info(f"Trained for {epoch_num + 1:02d} epochs, Best R@5: {best_r5:.1f}")


def main():
    # Initial setup: parser, logging...
    args = parser.parse_arguments()
    start_time = datetime.now()
    args.save_dir = join("logs", args.save_dir, start_time.strftime('%Y-%m-%d_%H-%M-%S') + '_' + args.title)
    commons.setup_logging(args.save_dir)
    commons.make_deterministic(args.seed, speedup=False)  # speedup=False make results Reproducible
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    logging.info(f"Using {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs")

    # Creation of Datasets
    logging.debug(f"Loading dataset {args.dataset_name} from folder {args.datasets_folder}")

    triplets_ds = datasets_ws.TripletsDataset(args, args.datasets_folder, args.dataset_name, "train",
                                              args.negs_num_per_query)
    logging.info(f"Train query set: {triplets_ds}")

    val_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "val")
    logging.info(f"Val set: {val_ds}")

    test_ds = datasets_ws.BaseDataset(args, args.datasets_folder, args.dataset_name, "test")
    logging.info(f"Test set: {test_ds}")

    #### Initialize model
    model = network.GeoLocalizationNet(args)
    model = model.to(args.device) 
    model = torch.nn.DataParallel(model)

    #### Setup Optimizer and Loss
    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.001)

    criterion_triplet = nn.TripletMarginLoss(margin=args.margin, p=2, reduction="sum")

    #### Resume model, optimizer, and other training parameters
    if args.resume:
        model, optimizer, best_r5, start_epoch_num, not_improved_num = util.resume_train(args, model, optimizer)
        logging.info(f"Resuming from epoch {start_epoch_num} with best recall@5 {best_r5:.1f}")
    else:
        best_r5 = start_epoch_num = not_improved_num = 0

    if torch.cuda.device_count() >= 2:
        # When using more than 1GPU, use sync_batchnorm for torch.nn.DataParallel
        model = convert_model(model)
        model = model.cuda()

    # Add tensorboard monitor
    writer = SummaryWriter(log_dir=args.save_dir)

    # Train model on train set and validate model on validation set every train epoch
    train(args,
          model,
          train_set=triplets_ds, loss_fn=criterion_triplet, optimizer=optimizer, val_set=val_ds,
          writer=writer, start_epoch_num=start_epoch_num, best_r5=best_r5, not_improved_num=not_improved_num,
          show_mining_triplet_img=False,
          visual_mining_save_path=args.train_visual_save_path,
          show_inference_results=False,
          visual_val_save_path=args.val_visual_save_path)
    logging.info(f"Trained total in {str(datetime.now() - start_time)[:-7]}")

    #### Test best model on test set
    best_model_state_dict = torch.load(join(args.save_dir, "best_model.pth"))["model_state_dict"]
    model.load_state_dict(best_model_state_dict)

    recalls, recalls_str = test.test(args, test_ds, model, test_method=args.test_method,
                                     show_inference_results=False,
                                     save_path=args.test_visual_save_path)
    logging.info(f"Recalls on {test_ds}: {recalls_str}")
    # show test results in one graph
    for i in range(len(recalls)):
        writer.add_scalar('Test', recalls[i], i + 1)
    writer.close()


if __name__ == '__main__':
    main()
