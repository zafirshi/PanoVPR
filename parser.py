
import os
import torch
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Benchmarking Visual Geolocalization",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Training parameters
    parser.add_argument("--train_batch_size", type=int, default=2,
                        help="Number of triplets (query, pos, negs) in a batch. Each triplet consists of 12 images")
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (caching and testing)")
    parser.add_argument("--criterion", type=str, default='triplet', 
                        help='loss to be used')
    parser.add_argument("--margin", type=float, default=0.1,
                        help="margin for the triplet loss")
    parser.add_argument("--epochs_num", type=int, default=60,
                        help="number of epochs to train for")
    parser.add_argument("--patience", type=int, default=10, help="_")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--optim", type=str, default="adam", choices=["adam", "sgd"])
    parser.add_argument("--cache_refresh_rate", type=int, default=125,
                        help="How often to refresh cache, in number of queries")
    # MARK: fine-tune should focus on 
    parser.add_argument("--queries_per_epoch", type=int, default=500,
                        help="How many queries to consider for one epoch. Must be multiple of cache_refresh_rate")
    parser.add_argument("--negs_num_per_query", type=int, default=10,
                        help="How many negatives to consider per each query in the loss")
    parser.add_argument("--neg_samples_num", type=int, default=125,
                        help="How many negatives to use to compute the hardest ones")
    parser.add_argument("--mining", type=str, default="partial", choices=["partial", "full", "random"])
    # Model parameters
    parser.add_argument("--backbone", type=str, default="swin_tiny",
                        choices=["swin_tiny", "swin_small","convnext_tiny", "convnext_small"], help="_")
    parser.add_argument("--l2", type=str, default="before_pool", choices=["before_pool", "after_pool", "none"],
                        help="When (and if) to apply the l2 norm with shallow aggregation layers")
    parser.add_argument("--aggregation", type=str, default="gem", choices=["gem", "spoc", "mac", "rmac"])

    parser.add_argument('--pca_dim', type=int, default=None, help="PCA dimension (number of principal components). If None, PCA is not used.")
    parser.add_argument('--num_non_local', type=int, default=1, help="Num of non local blocks")
    parser.add_argument("--non_local", action='store_true', help="_")
    parser.add_argument('--channel_bottleneck', type=int, default=128, help="Channel bottleneck for Non-Local blocks")
    parser.add_argument('--fc_output_dim', type=int, default=None,
                        help="Output dimension of fully connected layer. If None, don't use a fully connected layer.")   
    parser.add_argument("--clip", type=int, default=10, choices=[1,3,5,7,10], help='Gradient clip avoiding loss wave')
    # Shift window parameters (in image or vector)
    parser.add_argument('--reduce_factor', type=int, default=1, help='/n -> window_stride shorten factor compared to args.feature')
    parser.add_argument('--split_nums', type=int, default=16, help='choose how many parts to split pano_datasets image')
    # Initialization parameters
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to load checkpoint from, for resuming training or testing.")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num_workers", type=int, default=6, help="num_workers for all dataloaders")
    parser.add_argument('--resize', type=int, default=[224, 224], nargs=2, help="Resizing shape for images (HxW).")
    parser.add_argument('--query_resize', type=int, default=[448, 448], nargs=2, help="Resize for query")
    parser.add_argument('--database_resize', type=int, default=[448, 3584], nargs=2, help="Resize for database")
    parser.add_argument('--test_method', type=str, default="hard_resize",
                        choices=["hard_resize", "single_query", "central_crop", "five_crops", "nearest_crop", "maj_voting"],
                        help="This includes pre/post-processing methods and prediction refinement")
    parser.add_argument("--val_positive_dist_threshold", type=int, default=25, help="_")
    parser.add_argument("--train_positives_dist_threshold", type=int, default=10, help="_")
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+",
                        help="Recalls to be computed, such as R@5.")
    # Data augmentation parameters
    parser.add_argument("--brightness", type=float, default=None, help="_")
    parser.add_argument("--contrast", type=float, default=None, help="_")
    parser.add_argument("--saturation", type=float, default=None, help="_")
    parser.add_argument("--hue", type=float, default=None, help="_")
    parser.add_argument("--rand_perspective", type=float, default=None, help="_")
    parser.add_argument("--horizontal_flip", action='store_true', help="_")
    parser.add_argument("--random_resized_crop", type=float, default=None, help="_")
    parser.add_argument("--random_rotation", type=float, default=None, help="_")
    # Paths parameters
    parser.add_argument("--datasets_folder", type=str, default='/home/shize/Datasets',
                        help="Path with all datasets")
    parser.add_argument("--dataset_name", type=str, default="pitts250k", help="Relative path of the dataset")
    parser.add_argument("--pca_dataset_folder", type=str, default=None,
                        help="Path with images to be used to compute PCA (ie: pitts30k/images/train")
    parser.add_argument("--save_dir", type=str, default="SwinT",
                        help="Folder name of the current run (saved in ./logs})")
    parser.add_argument("--title", type=str, required=True, help="Abstract the experiment config")
    parser.add_argument("--train_visual_save_path", type=str, default='visualize/Insert_x32w/train_set/',
                        help="Mining [train] results save path")
    parser.add_argument("--val_visual_save_path", type=str, default='visualize/Insert_x32w/val_set/',
                        help="Inference [val] results save path")
    parser.add_argument("--test_visual_save_path", type=str, default='visualize/Insert_x32w/test_set/',
                        help="Inference [test] results save path")
    args = parser.parse_args()

    if args.datasets_folder == None:
        try:
            args.datasets_folder = os.environ['DATASETS_FOLDER']
        except KeyError:
            raise Exception("You should set the parameter --datasets_folder or export " +
                            "the DATASETS_FOLDER environment variable as such \n" +
                            "export DATASETS_FOLDER=../datasets_vg/datasets")

    if args.queries_per_epoch % args.cache_refresh_rate != 0:
        raise ValueError("Ensure that queries_per_epoch is divisible by cache_refresh_rate, " +
                         f"because {args.queries_per_epoch} is not divisible by {args.cache_refresh_rate}")

    if args.pca_dim != None and args.pca_dataset_folder == None:
        raise ValueError("Please specify --pca_dataset_folder when using pca")

    if args.split_nums < 8:
        raise ValueError("split_nums should be specified to 8/16/24/32")

    return args

