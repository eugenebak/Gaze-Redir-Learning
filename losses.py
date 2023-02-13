import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import torch.autograd as autograd


class PerceptualLoss(torch.nn.Module):
    def __init__(self, model='lpips', net='alex', spatial=False, use_gpu=False, gpu_ids=[0], version='0.1'): # VGG using our perceptually-learned weights (LPIPS metric)
    # def __init__(self, model='net', net='vgg', use_gpu=True): # "default" way of using VGG as a perceptual loss
        super(PerceptualLoss, self).__init__()
        print('Setting up Perceptual loss...')
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.gpu_ids = gpu_ids
        self.model = dist_model.DistModel()
        self.model.initialize(model=model, net=net, use_gpu=use_gpu, spatial=self.spatial, gpu_ids=gpu_ids, version=version)
        print('...[%s] initialized'%self.model.name())
        print('...Done')

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """

        if normalize:
            target = 2 * target  - 1
            pred = 2 * pred  - 1

        return self.model.forward(target, pred)


def discriminator_loss(real, fake):
    GANLoss = nn.BCEWithLogitsLoss(reduction='mean')
    real_size = list(real.size())
    fake_size = list(fake.size())
    device = real.get_device()
    real_label = torch.zeros(real_size, dtype=torch.float32).to(device)
    fake_label = torch.ones(fake_size, dtype=torch.float32).to(device)

    discriminator_loss = (GANLoss(fake, fake_label) + GANLoss(real, real_label)) / 2

    return discriminator_loss


def generator_loss(fake):
    GANLoss = nn.BCEWithLogitsLoss(reduction='mean')
    fake_size = list(fake.size())
    device = fake.get_device()
    fake_label = torch.zeros(fake_size, dtype=torch.float32).to(device)
    return GANLoss(fake, fake_label)


def reconstruction_l1_loss(x, x_hat):
    loss_fn = nn.L1Loss(reduction='mean')
    return loss_fn(x.detach(), x_hat)


def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, eps=1e-6)
    sim = F.hardtanh(sim, -1.0, 1.0)
    return torch.acos(sim) * (180 / np.pi)


def pitchyaw_to_vector(pitchyaws):
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], 1)


def gaze_angular_loss(y, y_hat):
    y = pitchyaw_to_vector(y)
    y_hat = pitchyaw_to_vector(y_hat)
    loss = nn_angular_distance(y, y_hat)
    return torch.mean(loss)


def triplet_loss(a, p, n, margin=1.0):
    #TripletLoss= nn.TripletMarginWithDistanceLoss(distance_function=lamb da x, y: 1.0 - F.cosine_similarity(x, y), margin=2.0)
    #loss = TripletLoss(a, p, n)
    
    #SimTripletLoss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: F.cosine_similarity(x, y), margin=0.99)
    #loss = SimTripletLoss(a, n, p) 
    
    a = F.normalize(a, dim=-1)
    p = F.normalize(p, dim=-1)
    n = F.normalize(n, dim=-1)
    TripletLoss = nn.TripletMarginLoss(margin=margin, p=2.0, eps=1e-06)
    loss = TripletLoss(a, p , n)
    return loss


def contrastive_loss(p, n, margin = 1.0):
    #margin_dist = 2.0 - (1.0 - F.cosine_similarity(p, n))
    #sim_contrastive_loss = F.cosine_similarity(p, n) - margin
    #loss = F.relu(sim_contrastive_loss)
    
    p = F.normalize(p, dim=-1)
    n = F.normalize(n, dim=-1)
    L2Distance = nn.PairwiseDistance(p=2.0, eps=1e-06, keepdim=False)
    loss = margin - L2Distance(p, n)
    loss = 1/2 * torch.square(F.relu(loss))
    return torch.mean(loss)

