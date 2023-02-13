import os
import json
import random
import wandb
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
from torchvision import datasets
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset

import losses
from dataset import HDFDataset
from utils import save_images, worker_init_fn, send_data_dict_to_gpu, recover_images, def_test_list, RunningStatistics,\
    adjust_learning_rate, script_init_common, get_example_images, save_model, load_model, rotate, rotation_matrix_1d, \
    rotation_matrix_2d, extract_eye_feature

from models.encoder import Encoder
from models.decoder import Decoder

from models.eyeresnet import EyeFeatureExtractor, Eyeresnet

parser = argparse.ArgumentParser()

# Training parameter
parser.add_argument("-n_epochs", type=int, default=3, help="number of epochs of training")
parser.add_argument('-bsz', type=int, default=256, metavar='', help='Batch size')
parser.add_argument('-eval_bsz', type=int, default=1024, metavar='', help='Batch size')
parser.add_argument('-num_workers', type=int, default=2, help='Number of workers')
parser.add_argument('-seed', type=int, default=1, help='Random seed')

parser.add_argument('-gazecapture_path', type=str, default='/home/cvip1/nas/backup1/personal_psj/Gaze/faze_preprocess/outputs_faze/GazeCapture_eye.h5', metavar='')
parser.add_argument("-train_subsample", type=float, default=0.3, help="Subsample ratio for sampling the train dataset")
parser.add_argument("-test_subsample", type=float, default=0.0025, help="Subsample ratio for sampling the test dataset")

parser.add_argument('-save_freq', type=int, default=500, metavar='', help='Frequency to print training process')
parser.add_argument("-lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("-b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("-b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("-weight_decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("-gpu", type=int, default=0, help="GPU ID")

# Log argument
parser.add_argument("-project_name", type=str, default="eyenet", help="project name for wandb")
parser.add_argument("-exp_name", type=str, default="base", help="Experiment name for wandb")
parser.add_argument('-print_freq', type=int, default=20, metavar='', help='Frequency to print training process')
parser.add_argument("-sample_interval", type=int, default=500, help="Sample download ")

# Result Path
parser.add_argument('-result_path', type=str, default='./result', metavar='', help='Random seed')

# Checkpoints
parser.add_argument('-load_step', type=int, default=0 , help='Checkpoint step')

# Loss function

args = parser.parse_args()
print(args)

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"] ="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import gc
gc.collect()
torch.cuda.empty_cache()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Result path
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)
    
result_path = os.path.join(args.result_path, args.project_name, args.exp_name)
if not os.path.exists(result_path):
    os.makedirs(result_path)

# Wandb setting
os.environ['WANDB_API_KEY'] = "0ee23525f6f4ddbbab74086ddc0b2294c7793e80"
wandb.init(project=args.project_name, entity="psj", name=args.exp_name)
wandb.config.update(args)

import logging
logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

# Read GazeCapture train/val/test split
with open('./gazecapture_split.json', 'r') as f:
    all_gc_prefixes = json.load(f)

all_data = OrderedDict()

train_prefixes = all_gc_prefixes["train"]
train_dataset = HDFDataset(hdf_file_path=args.gazecapture_path, prefixes=train_prefixes,
                           is_bgr=False, get_2nd_sample=False)
if args.train_subsample < 1.0:
    train_dataset = Subset(train_dataset, 
                           np.linspace(start=0, stop=len(train_dataset), num=int(args.train_subsample * len(train_dataset)), endpoint=False, dtype=np.uint32))
train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)

all_data["gc/train"] = {"dataset": train_dataset, "loader": train_loader}
for tag, hdf_file, is_bgr, prefixes in [
        ("gc/val", args.gazecapture_path, False, all_gc_prefixes["val"]),
        ("gc/test", args.gazecapture_path, False, all_gc_prefixes["test"])
        ]:
    dataset = HDFDataset(hdf_file_path=hdf_file, prefixes=prefixes, is_bgr=is_bgr, get_2nd_sample=True, pick_at_least_per_person=2)
    
    if tag == "gc/val":
        if args.test_subsample < 1.0:
            dataset = Subset(dataset, 
                             np.linspace(start=0, stop=len(dataset), num=int(args.test_subsample * len(dataset)),
                                         endpoint=False, dtype=np.uint32))
    all_data[tag] = {
        "dataset": dataset,
        "loader": DataLoader(dataset, batch_size=args.eval_bsz, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        }

logging.info("")
for tag, val in all_data.items():
    tag = "[%s]" % tag
    dataset = val["dataset"]
    original_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
    num_people = len(original_dataset.prefixes)
    num_original_dataset = len(original_dataset)
    
    logging.info("%10s number of subjects:  %7d" % (tag, num_people))
    logging.info("%10s full set size:       %7d" % (tag, num_original_dataset))
    logging.info("%10s current set size:    %7d" % (tag, len(dataset)))
    logging.info("")

# Networks
model = Eyeresnet().to(device)

# Load the checkpoint
if args.load_step != 0:
    if args.load_step == -1:
        model.load_state_dict(torch.load(os.path.join(result_path, "model_final.pth")))
    else:
        model.load_state_dict(torch.load(os.path.join(result_path, "model_%d.pth" % args.load_step)))

losses_dict = OrderedDict()
val_dict = OrderedDict()

# Optimizers
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(args.n_epochs):
    for i, data in enumerate(all_data["gc/train"]["loader"]):
        model.train()
        iteration = epoch * len(train_loader) + i
        data = send_data_dict_to_gpu(data, device)
        
        y_pred = model(data["image_a"])
        
        losses_dict["loss"] = losses.gaze_angular_loss(data["gaze_a"], y_pred)
        total_loss = losses_dict["loss"]
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        wandb.log(losses_dict)
        
        # Print the training process
        if iteration % args.print_freq == 0:
            logging.info("[Epoch %d/%d] [Iteration %d/%d] %s" % (epoch, args.n_epochs, iteration, args.n_epochs * len(train_loader),
                                                            ' '.join(["[%s: %.5f]" % (k, v) for k, v in losses_dict.items()])
                                                            )
            )
        
        # Save the model every specified iteration.
        if iteration != 0 and iteration % args.save_freq == 0:
            ckpt_path = os.path.join(result_path, "model_%d.pth" % iteration)
            torch.save(model.state_dict(), ckpt_path)
        
        if iteration % args.sample_interval == 0:
            total_val = 0
            val_error = 0
            for idx, data in enumerate(all_data["gc/val"]["loader"]):
                with torch.no_grad():
                    model.eval()
                    data = send_data_dict_to_gpu(data, device)
                    total_val += data["image_a"].size(0)
                    y_pred = model(data["image_a"])
                    val_error += losses.gaze_angular_loss(data["gaze_a"], y_pred)
            val_dict["Validation error"] = val_error / total_val
            wandb.log(val_dict)

logging.info("Saving the final model...")
ckpt_path = os.path.join(result_path, "model_final.pth")
torch.save(model.state_dict(), ckpt_path)
logging.info("Training finished.")

logging.info("Start to evaluate.")
total_test = 0
test_error = 0
for idx, data in tqdm(enumerate(all_data["gc/test"]["loader"])):
    with torch.no_grad():
        data = send_data_dict_to_gpu(data, device)
        total_test += data["image_a"].size(0)
        y_pred = model(data["image_a"])
        test_error += losses.gaze_angular_loss(data["gaze_a"], y_pred)
logging.info("[Gaze error on GazeCapture test split] %.3f" % test_error / total_test)

gc.collect()
torch.cuda.empty_cache()


            