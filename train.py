import os
import json
import random
import wandb
import torch
import argparse
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchattacks
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

from models.gazeheadnet import GazeHeadNet
from models.gazeheadResnet import GazeHeadResNet
from models.discriminator import PatchGAN
from models.redirector import Redirector
from models.EyeFeatureExtractor import EyeFeatureExtractor, eyeresnet18

parser = argparse.ArgumentParser()

# Training parameter
parser.add_argument("-n_epochs", type=int, default=3, help="number of epochs of training")
parser.add_argument('-bsz', type=int, default=64, metavar='', help='Batch size')
parser.add_argument('-eval_bsz', type=int, default=128, metavar='', help='Batch size')
parser.add_argument('-num_workers', type=int, default=2, help='Number of workers')
parser.add_argument('-seed', type=int, default=1, help='Random seed')

parser.add_argument('-gazecaputre_path', type=str, default='', metavar='')
parser.add_argument("-train_subsample", type=float, default=0.15, help="Subsample ratio for sampling the train dataset")
parser.add_argument("-test_subsample", type=float, default=0.025, help="Subsample ratio for sampling the test dataset")

parser.add_argument('-save_freq', type=int, default=1000, metavar='', help='Frequency to print training process')
parser.add_argument("-lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("-b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("-b2", type=float, default=0.999, help="adam: decay of second order momentum of gradient")
parser.add_argument("-weight_decay", type=float, default=1e-4, help="Weight decay")
parser.add_argument("-gpu", type=int, default=0, help="GPU ID")

# Log argument
parser.add_argument("-project_name", type=str, default="purifier_cifar10", help="project name for wandb")
parser.add_argument("-exp_name", type=str, default="base", help="Experiment name for wandb")
parser.add_argument('-print_freq', type=int, default=20, metavar='', help='Frequency to print training process')
parser.add_argument("-sample_interval", type=int, default=100, help="Sample download ")

# Result Path
parser.add_argument('-result_path', type=str, default='./result', metavar='', help='Random seed')

# Checkpoints
parser.add_argument('-load_step', type=int, default=0 , help='Checkpoint step')

# Loss function
parser.add_argument("-coeff_gan", type=float, default=1.0)
parser.add_argument("-coeff_embedding", type=float, default=1.0)
parser.add_argument("-coeff_pl", type=float, default=2.0)
parser.add_argument("-coeff_gaze", type=float, default=5.0)
parser.add_argument("-coeff_l1", type=float, default=200.0)
parser.add_argument("-coeff_percep", type=float, default=200.0)
parser.add_argument("-coeff_redirec", type=float, default=5.0)
parser.add_argument("-coeff_cns", type=float, default=1.0)
parser.add_argument("-coeff_sg", type=float, default=1.0)
parser.add_argument("-coeff_consistency", type=float, default=1.0)

 
parser.add_argument("-contrast_margin", type=float, default=1.0)

# Network configuration
parser.add_argument("-densenet_blocks", type=int, default=5)
parser.add_argument("-growth_rate", type=int, default=32)
# =============================================================================
# Check
# =============================================================================
parser.add_argument("-num_1d_units", type=int, default=0)
parser.add_argument("-num_2d_units", type=int, default=8)
parser.add_argument("-size_0d_unit", type=int, default=1024)
parser.add_argument("-size_1d_unit", type=int, default=16)
parser.add_argument("-size_2d_unit", type=int, default=16)

# Pretrained weights path
parser.add_argument("-eval_gazenet", type=str, default="./models/weights/eval_gazenet.tar")
parser.add_argument("-gazenet", type=str, default="./models/weights/gazenet.tar")

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
                           is_bgr=False, get_2nd_sample=True, num_labeld_samples=100)
train_loader = DataLoder(train_dataset, batch_size=args.bsz, shuffle=True, drop_last=True, num_workers=args.num_workers, pin_memory=True)

all_data["gc/train"] = {"dataset": train_dataset, "loader": train_loader}
for tag, hdf_file, is_bgr, prefixes in [
        ("gc/val", args.gazecapture_path, False, all_gc_prefixes["val"]),
        ("gc/test", args.gazecapture_path, False, all_gc_prefixes["test"]),
        ("mpi", args.mpiigaze_path, False, None)]:
    dataset = HDFDataset(hdf_file_path=hdf_file, prefixes=prefixes, is_bgr=is_bgr, get_2nd_sample=True, pick_at_least_per_person=2)
    
    if tag == "gc/test":
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
configuration = []
if args.size_0d_unit > 0:
    configuration += [(0, args.size_0d_unit)]
configuration += ([(1, args.size_1d_unit)] * args.num_1d_units +
                  [(2, args.size_2d_unit)] * args.num_2d_units)
num_all_pseudo_labels = np.sum([dof for dof, _ in configuration])
num_all_embedding_features = np.sum([(dof + 1) * num_feats for dof, num_feats in configuration])

encoder = Encoder(num_all_pseudo_labels, num_all_embedding_features, configuration).to(device)
decoder = Decoder(num_all_embedding_features).to(device)

eye_extractor = EyeFeatureExtractor().to(device)

discriminator = PatchGAN(input_nc=3).to(device)



# Load the checkpoint
if args.load_step != 0:
    if args.load_step == -1:
        purifier.load_state_dict(torch.load(os.path.join(result_path, "model_final.pth")))
    else:
        purifier.load_state_dict(torch.load(os.path.join(result_path, "model_%d.pth" % args.load_step)))

# Loss function
lpips = PerceptualLoss(model='lpips', net='alex').to(device)

# Pretrained_network
GazeHeadNet_train = GazeHeadNet().to(device)
GazeHeadNet_eval = GazeHeadResNet().to(device)
eyenet = eyeresnet18().to(device)

GazeHeadNet_eval.load_state_dict(torch.load(args.eval_gazenet))
GazeHeadNet_train.load_state_dict(torch.load(args.gazenet))
# eyenet.load_state_dict(torch.load(config.gazenet_savepath)) # No weight !!!!!!!!!!!

for param in GazeHeadNet_train.parameters():
    param.requires_grad = False
for param in GazeHeadNet_eval.parameters():
    param.requires_grad = False
for param in lpips.parameters():
    param.requires_grad = False
for param in eyenet.parameters():
    param.requires_grad = False

losses_dict = OrderedDict()
param_dict = OrderedDict()

# Optimizers
generator_params = [encoder.parameters(), decoder.parameters()]
generator_optimizer = optim.Adam(generator_params, lr=args.lr, weight_decay=args.weight_decay)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
eye_extractor_optimizer = optim.Adam(eye_extractor.parameters(), lr=config.lr, weight_decay=config.l2_reg)

alpha_sim = torch.full((32,), 0.5, dtype=torch.float).to(device)

# ----------
#  Training
# ----------

"""
input_dict = image_a, gaze_a, head_b, (image_b, gaze_b, head_b)
output_dict = image_b_hat, image_b_hat_all, embeds, pls
"""
for epoch in range(args.n_epochs):
    for i, data in enumerate(train_loader):
        iteration = epoch * len(train_loader) + i
        
        encoder.train()
        decoder.train()
        eye_extractor.train()
        
        data = send_data_dict_to_gpu(data, device)
        
        embeddings_a, pseudo_labels_a = encoder(data['image_a'])
        embeddings_b, pseudo_labels_b = encoder(data['image_b'])
        
        # =============================================================================
        # Mix the features
        # =============================================================================
        num_0d_units = 1 if args.size_0d_unit > 0 else 0
        random = np.random.randint(2, size=[num_0d_units + args.num_1d_units + args.num_2d_units,
                                            args.bsz, 1, 1]).tolist()
        random_tensor = torch.tensor(random, dtype=torch.float, requires_grad=False).to(device)

        normalized_embeddings_from_a_mix = rotate(embeddings_a, pseudo_labels_a, random_tensor, inverse=True)
        embeddings_a_to_mix = rotate(normalized_embeddings_from_a_mix, pseudo_labels_b, random_tensor)
        
        decoded_embeddings_a_to_mix = decoder(embeddings_a_to_mix)
        embeddings_mix_hat, pseudo_labels_mix_hat = encoder(decoded_embeddings_a_to_mix)
        # =============================================================================
        
        # =============================================================================
        # source to target
        # =============================================================================
        normalized_embeddings_from_a = rotate(embeddings_a, pseudo_labels_a, inverse=True)
        embeddings_a_to_b = rotate(normalized_embeddings_from_a, pseudo_labels_b)
        
        image_b_hat = decoder(embeddings_a_to_b)
        # =============================================================================
        
        # optimize discriminator
        real = discriminator(data['image_b'])
        fake = discriminator(image_b_hat.detach())
        
        losses_dict['discriminator'] = losses.discriminator_loss(real=real, fake=fake)
        losses_dict['generator'] = losses.generator_loss(fake=fake)
        discriminator_loss = losses_dict['discriminator'] * args.coeff_gan
        # Warm up period for generator losses
        losses_dict['discrim_coeff'] = torch.tensor(max(min(1.0, current_step / 20000.0), 0.0))
        
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        for param in discriminator.parameters():
            param.requires_grad = False
        
        losses_dict['l1'] = losses.reconstruction_l1_loss(x=data['image_b'], x_hat=image_b_hat)
        total_loss = losses_dict['l1'] * config.coeff_l1
        
        # =============================================================================
        # Gaze hardness aware transformation (GHT)
        # =============================================================================
        bsz = embeddings_a[-1].shape[0]
        z_s_g = embeddings_a[-1].reshape(bsz, -1)
        z_t_g = embeddings_b[-1].reshape(bsz, -1)
        
        g_s = pseudo_labels_a[-1]
        g_t = pseudo_labels_b[-1]
        
        z_tr_g = alpha_sim * z_s_g + (1 - alpha_sim) * z_t_g
        g_tr = alpha_sim * g_s + (1 - alpha_sim) * g_t
        
        rotate = lambda feature, label_a, label_b: torch.matmul(rotation_matrix_2d(label_b, False), torch.matmul(
                        rotation_matrix_2d(label_a, True), feature.reshape(args.bsz, 3, -1))).reshape(args.bsz, -1)
        z_tr_g_tilde = rotate(z_tr_g, g_tr, g_t)
        z_t_g_tilde = rotate(z_s_g, g_s, g_t)
        
        # Consistency loss
        alpha_sim = 1 - F.cosine_similarity(z_tr_g_tilde, z_t_g_tilde, eps=1e-6, dim=-1)
        losses_dict['cns'] = torch.mean(1 - F.cosine_similarity(z_tr_g_tilde, z_t_g_tilde, eps=1e-6, dim=-1))
        total_loss += losses_dict['cns'] * args.coeff_cns
        # =============================================================================
        
        # =============================================================================
        # Structured Gaze loss (SG loss)
        # =============================================================================
        z_e = extract_eye_feature(data['image_a']).reshape(bsz, -1)
        
        alpha = np.random.beta(2.0, 2.0, size=(bsz,))
        alpha = torch.from_numpy(alpha).float().cuda()
        alpha.requires_grad = False
        
        z_s_p = (alpha + 1) * z_s_g - alpha * z_s_e # Positive extrapolation
        z_s_p = F.normalize(z_s_p)
        
        z_s_h = embeddings_a[-2].reshape(bsz, -1)
        z_s_u1 = embeddings_a[-3].reshape(bsz, -1)
        z_s_u2 = embeddings_a[-4].reshape(bsz, -1)
        
        inssential = (z_s_h, z_s_u1, z_s_u2)
        neg_features = [z_s_h, z_s_u1, z_s_u2]
        for z_s_i in inessential:
            z_diff = z_s_g - z_s_i
            z_diff = F.normalize(z_diff)
            k = torch.bmm(z_s_g.view(bsz, 1, -1), z_diff.view(bsz, -1, 1))
            
            alpha_0 = k + (1 - k) * alpha
            z_s_n = alpha_0 * z_s_g + (1 - alpha_0) * z_s_i
            neg_features.append(z_s_n)
        
        pos_features = [z_s_g, z_s_p]
        
        losses_dict['sg'] = 0
        sg_count = 0
        for pos in pos_features:
            for neg in neg_features:
                sg_count += 1
                losses_dict['sg'] += losses.triplet_loss(eye_feature, pos, neg, margin=args.contrast_margin)
        
        losses_dict['sg'] /= sg_count
        total_loss += losses_dict['sg'] * config.coeff_sg
        # =============================================================================
        
        losses_dict['gaze_a'] = (losses.gaze_angular_loss(y=data['gaze_a'], y_hat=pseudo_labels_a[-1]) +
                                 losses.gaze_angular_loss(y=data['gaze_b'], y_hat=pseudo_labels_b[-1]))/2
        losses_dict['head_a'] = (losses.gaze_angular_loss(y=data['head_a'], y_hat=pseudo_labels_a[-2]) +
                                 losses.gaze_angular_loss(y=data['head_b'], y_hat=pseudo_labels_b[-2]))/2
        
        total_loss += (losses_dict['gaze_a'] + losses_dict['head_a']) * args.coeff_gaze
        
        fake = discriminator(image_b_hat)
        generator_loss = losses.generator_loss(fake=fake)
        total_loss += generator_loss * args.coeff_gan * losses_dict['discrim_coeff']
        
        if args.coeff_consistency != 0:
            normalized_embeddings_from_a = rotate(embeddings_a, pseudo_labels_a, inverse=True)
            normalized_embeddings_from_b = rotate(embeddings_b, pseudo_labels_b, inverse=True)
            
            flattened_normalized_embeddings_from_a = torch.cat([
                e.reshape(e.shape[0], -1) for e in normalized_embeddings_from_a], dim=1)
            flattened_normalized_embeddings_from_b = torch.cat([
                e.reshape(e.shape[0], -1) for e in normalized_embeddings_from_b], dim=1)
            losses_dict['embedding_consistency'] = (1.0 - torch.mean(
                F.cosine_similarity(flattened_normalized_embeddings_from_a,
                                    flattened_normalized_embeddings_from_b, dim=-1)))
            
            total_loss += losses_dict['embedding_consistency'] * args.coeff_consistency
        
        if args.coeff_embedding != 0:
            flattened_before_c = torch.cat([
                e.reshape(e.shape[0], -1) for e in embeddings_a_to_mix], dim=1)
            flattened_after_c = torch.cat([
                e.reshape(e.shape[0], -1) for e in embeddings_mix_hat], dim=1)
            
            losses_dict['embedding_disentangle'] = (1.0 - torch.mean(
                F.cosine_similarity(flattened_before_c,
                                    flattened_after_c, dim=-1)))
            total_loss += losses_dict['embedding_disentangle'] * args.coeff_disentangle_embedding_loss
            
        if args.coeff_pl != 0:
            losses_dict['label_disentangle'] = 0
            pseudo_labels_a_b_mix = []
            for i in range(len(pseudo_labels_a)):  # pseudo code
                if pseudo_labels_b[i] is not None:
                    pseudo_labels_a_b_mix.append(
                        pseudo_labels_b[i] * random_tensor[i].squeeze(-1) + pseudo_labels_a[i] * (1 - random_tensor[i].squeeze(-1)))
                else:
                    pseudo_labels_a_b_mix.append(None)

            for y, y_hat in zip(pseudo_labels_a_b_mix[-2:], pseudo_labels_mix_hat[-2:]):
                if y is not None:
                    losses_dict['label_disentangle'] += losses.gaze_angular_loss(y, y_hat)
            total_loss += losses_dict['label_disentangle'] * args.coeff_pl
            
        feature_h, gaze_h, head_h = GazeHeadNet_train(image_b_hat, True)
        feature_t, gaze_t, head_t = GazeHeadNet_train(data['image_b'], True)
        losses_dict['redirection_feature_loss'] = 0
        
        for i in range(len(feature_h)):
            losses_dict['redirection_feature_loss'] += nn.functional.mse_loss(feature_h[i], feature_t[i].detach())
        total_loss += losses_dict['redirection_feature_loss'] * args.coeff_percep
        losses_dict['gaze_redirection'] = losses.gaze_angular_loss(y=gaze_t.detach(), y_hat=gaze_h)
        total_loss += losses_dict['gaze_redirection'] * args.coeff_redirec
        losses_dict['head_redirection'] = losses.gaze_angular_loss(y=head_t.detach(), y_hat=head_h)
        total_loss += losses_dict['head_redirection'] * args.coeff_redirec
        
        generator_optimizer.zero_grad()
        eye_extractor_optimizer.zero_grad()
        total_loss.backward()
        generator_optimizer.step()
        eye_extractor_optimizer.step()
        
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
            models = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "discriminator": discriminator.state_dict(),
                "eye_extractor": eye_extractor.state_dict(),
                }
            torch.save(models, ckpt_path)
        
        if iteration % args.sample_interval == 0:
            total_test = 0
            
            test_dict = OrderedDict()
            test_dict["gaze_a"] = 0
            test_dict["head_a"] = 0
            test_dict["head_to_gaze"] = 0
            test_dict["gaze_to_head"] = 0
            test_dict["u_to_gaze"] = 0
            test_dict["u_to_head"] = 0
            test_dict["head_redirection"] = 0
            test_dict["gaze_redirection"] = 0
            test_dict["lpips"] = 0
            test_dict["l1"] = 0
            test_dict["lpips_all"] = 0
            test_dict["l1_all"] = 0
            
            for idx, data in enumerate(test_loader):
                total_test += data["image_a"].size(0)
                embeddings_a, pseudo_labels_a = encoder(data['image_a'])
                
                losses_dict['gaze_a'] += losses.gaze_angular_loss(y=data['gaze_a'], y_hat=pseudo_labels_a[-1])
                losses_dict['head_a'] += losses.gaze_angular_loss(y=data['head_a'], y_hat=pseudo_labels_a[-2])
                
                image_a_rec = decoder(embeddings_a)
                gaze_a_rec, head_a_rec = GazeHeadNet_eval(image_a_rec)
                
                # embedding disentanglement error
                idx = 0
                gaze_disentangle_loss = 0
                head_disentangle_loss = 0
                batch_size = data['image_a'].shape[0]
                num_0d_units = 1 if args.size_0d_unit > 0 else 0
                for dof, num_feats in configuration:
                    if dof != 0:
                        random_angle = (torch.rand(batch_size, dof).to(device) - 0.5) * np.pi * 0.2
                        random_angle += pseudo_labels_a[idx]
                        if dof == 2:
                            rotated_embedding = torch.matmul(rotation_matrix_2d(random_angle, False), torch.matmul(
                                rotation_matrix_2d(pseudo_labels_a[idx], True), embeddings_a[idx]))
                        else:
                            rotated_embedding = torch.matmul(rotation_matrix_1d(random_angle, False), torch.matmul(
                                rotation_matrix_1d(pseudo_labels_a[idx], True), embeddings_a[idx]))
                        new_embedding = [item for item in embeddings_a]
                        new_embedding[idx] = rotated_embedding
                        
                        image_random = decoder(new_embedding)
                        gaze_random, head_random = GazeHeadNet_eval(image_random)
                        if idx < args.num_1d_units + args.num_2d_units + num_0d_units - 2:
                            gaze_disentangle_loss += losses.gaze_angular_loss(gaze_a_rec, gaze_random)
                            head_disentangle_loss += losses.gaze_angular_loss(head_a_rec, head_random)
                        if idx == args.num_1d_units + args.num_2d_units + num_0d_units - 2:  # head
                            losses_dict['head_to_gaze'] += losses.gaze_angular_loss(gaze_a_rec, gaze_random)
                        if idx == args.num_1d_units + args.num_2d_units + num_0d_units - 1:  # gaze
                            losses_dict['gaze_to_head'] += losses.gaze_angular_loss(head_a_rec, head_random)
                    idx += 1
                if config.num_1d_units + config.num_2d_units - 2 != 0:
                    losses_dict['u_to_gaze'] += gaze_disentangle_loss / (
                                config.num_1d_units + config.num_2d_units - 2)
                    losses_dict['u_to_head'] += head_disentangle_loss / (
                                config.num_1d_units + config.num_2d_units - 2)
                
                # Calculate some errors if target image is available
                if 'image_b' in data:
                    # redirect with pseudo-labels
                    gaze_embedding = torch.matmul(rotation_matrix_2d(data['gaze_b'], False),
                                                  torch.matmul(rotation_matrix_2d(pseudo_labels_a[-1], True),
                                                               embeddings_a[-1]))
                    head_embedding = torch.matmul(rotation_matrix_2d(data['head_b'], False),
                                                  torch.matmul(rotation_matrix_2d(pseudo_labels_a[-2], True),
                                                               embeddings_a[-2]))
                    embeddings_a_to_b = embeddings_a[:-2]
                    embeddings_a_to_b.append(head_embedding)
                    embeddings_a_to_b.append(gaze_embedding)
                    
                    output_dict['image_b_hat'] = decoder(embeddings_a_to_b)
                    
                    gaze_b_hat, head_b_hat = GazeHeadNet_eval(output_dict['image_b_hat'])
                    losses_dict['head_redirection'] += losses.gaze_angular_loss(y=data['head_b'], y_hat=head_b_hat)
                    losses_dict['gaze_redirection'] += losses.gaze_angular_loss(y=data['gaze_b'], y_hat=gaze_b_hat)
                    
                    losses_dict['lpips'] += torch.mean(lpips(data['image_b'], output_dict['image_b_hat']))
                    losses_dict['l1'] += losses.reconstruction_l1_loss(data['image_b'], output_dict['image_b_hat'])
                
                    embeddings_b, pseudo_labels_b = encoder(data['image_b'])
                    
                    normalized_embeddings_from_a = rotate(embeddings_a, pseudo_labels_a, inverse=True)
                    embeddings_a_to_b_all = rotate(normalized_embeddings_from_a, pseudo_labels_b)
                    
                    output_dict['image_b_hat_all'] = decoder(embeddings_a_to_b_all)
                    
                    losses_dict['lpips_all'] += torch.mean(lpips(data['image_b'], output_dict['image_b_hat_all']))
                    losses_dict['l1_all'] += losses.reconstruction_l1_loss(data['image_b'], output_dict['image_b_hat_all'])
                
            for k, v in test_dict.items():
                test_dict[k] = v / total_test
        
        torch.cuda.empty_cache()
        
    G_scheduler.step()
    D_scheduler.step()

gc.collect()
torch.cuda.empty_cache()

logging.info("Saving the final model...")
ckpt_path = os.path.join(result_path, "model_final.pth")
models = {
    "encoder": encoder.state_dict(),
    "decoder": decoder.state_dict(),
    "discriminator": discriminator.state_dict(),
    "eye_extractor": eye_extractor.state_dict(),
    }
torch.save(models, ckpt_path)
logging.info("Training finished.")
            