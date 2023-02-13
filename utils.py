"""

author: san9569@naver.com (Sang Jin Park)
"""

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def recover_image(x, cv=True):
    # x = (x + 1.0) * (255.0 / 2.0)
    x = x.detach().cpu().numpy()
    x = x * 255.0
    x = np.clip(x, 0, 255)
    x = x.astype(np.uint8)
    if len(x.shape) == 4:
        x = np.transpose(x, [0, 2, 3, 1])
        if cv:
            x = x[:, :, :, ::-1]
    else:
        x = np.transpose(x, [1, 2, 0])
        if cv:
            x = x[:, :, ::-1]
    return x


def rotation_matrix_1d(self, pseudo_label, inverse=False):
        cos = torch.cos(pseudo_label)
        sin = torch.sin(pseudo_label)
        matrices = torch.stack([cos, -sin, sin, cos], dim=1).view(-1, 2, 2)
        if inverse:
            matrices = torch.transpose(matrices, 1, 2)
        return matrices

rot2d_ones = None
rot2d_zeros = None

def rotation_matrix_2d(self, pseudo_label, inverse=False):
    cos = torch.cos(pseudo_label)
    sin = torch.sin(pseudo_label)
    if (self.rot2d_ones is None
            or self.rot2d_zeros is None
            or self.rot2d_ones.shape[0] != pseudo_label.shape[0]):
        self.rot2d_ones = torch.ones_like(cos[:, 0])
        self.rot2d_zeros = torch.zeros_like(cos[:, 0])
    ones = self.rot2d_ones
    zeros = self.rot2d_zeros
    matrices_1 = torch.stack([ones, zeros, zeros,
                              zeros, cos[:, 0], -sin[:, 0],
                              zeros, sin[:, 0], cos[:, 0]
                              ], dim=1)
    matrices_2 = torch.stack([cos[:, 1], zeros, sin[:, 1],
                              zeros, ones, zeros,
                              -sin[:, 1], zeros, cos[:, 1]
                              ], dim=1)
    matrices_1 = matrices_1.view(-1, 3, 3)
    matrices_2 = matrices_2.view(-1, 3, 3)
    matrices = torch.matmul(matrices_2, matrices_1)
    if inverse:
        matrices = torch.transpose(matrices, 1, 2)
    return matrices


def rotate(self, embeddings, pseudo_labels, rotate_or_not=None, inverse=False):
    rotation_matrices = []
    for (dof, _), pseudo_label in zip(self.configuration, pseudo_labels):
        if dof == 0:
            assert pseudo_label is None
            rotation_matrix = None
        elif dof == 1:
            rotation_matrix = self.rotation_matrix_1d(pseudo_label, inverse=inverse)
        elif dof == 2:
            rotation_matrix = self.rotation_matrix_2d(pseudo_label, inverse=inverse)
        else:
            raise ValueError('Invalid DoF: %d' % dof)
        rotation_matrices.append(rotation_matrix)
    rotated_embeddings = [
        torch.matmul(rotation_matrix, embedding)
        if rotation_matrix is not None else embedding
        for embedding, rotation_matrix in zip(embeddings, rotation_matrices)
    ]

    if rotate_or_not is not None:
        mixed_embeddings = [rotated_embeddings[i] * rotate_or_not[i] + embeddings[i] * (1 - rotate_or_not[i])
                            for i in range(len(embeddings))]
        return mixed_embeddings
    return rotated_embeddings


def extract_eye_feature(self, x):
    eye_images = crop_eye_img(x)
    eye_feature = self.eyenet(eye_images)
    z_eye = self.eye_extractor(eye_feature)
    z_eye = z_eye.reshape(x.shape[0], 3, -1)
    return z_eye


def worker_init_fn(worker_id):
    # Custom worker init to not repeat pairs
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def send_data_dict_to_gpu(data, device):
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    return data


def save_images(save_path, walks, keys, cycle=True):
    num_walks_per_sample = len(walks)
    num_img_per_walk = len(walks[0])
    num_samples = walks[0][0].shape[0]
    size = walks[0][0].shape[2:4]
    size = size[::-1]
    for i in range(num_samples):
        path = os.path.join(save_path, str(keys[i]))
        if not os.path.isdir(path):
            os.makedirs(path)
        for j in range(num_walks_per_sample):
            frames = [recover_images(walks[j][k][i]) for k in range(num_img_per_walk)]
            if cycle:
                frames += frames[::-1]  # continue in reverse
            clip = cv2.VideoWriter('%s/%s.avi' % (path, str(j)), cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'), 20, size, True)
            for k in range(len(frames)):
                clip.write(frames[k])
            clip.release()


def recover_images(x):
    # Every specified iterations save sample images
    # Note: We're doing this separate to Tensorboard to control which input
    #       samples we visualize, and also because Tensorboard is an inefficient
    #       way to store such images.
    x = x.cpu().numpy()
    x = (x + 1.0) * (255.0 / 2.0)
    x = np.clip(x, 0, 255)  # Avoid artifacts due to slight under/overflow
    x = x.astype(np.uint8)
    if len(x.shape) == 4:
        x = np.transpose(x, [0, 2, 3, 1])  # CHW to HWC
        x = x[:, :, :, ::-1]  # RGB to BGR for OpenCV
    else:
        x = np.transpose(x, [1, 2, 0])  # CHW to HWC
        x = x[:, :, ::-1]  # RGB to BGR for OpenCV
    return x


def def_test_list():
    # fixed visualization pairs from GazeCapture test set
    test_list = []
    test_list.append({'key': '01200', 'idx_a': 406, 'idx_b': 430})
    test_list.append({'key': '01370', 'idx_a': 1248, 'idx_b': 431})
    test_list.append({'key': '01376', 'idx_a': 136, 'idx_b': 487})
    test_list.append({'key': '01425', 'idx_a': 463, 'idx_b': 269})
    test_list.append({'key': '01517', 'idx_a': 893, 'idx_b': 661})
    test_list.append({'key': '01525', 'idx_a': 914, 'idx_b': 542})
    test_list.append({'key': '01575', 'idx_a': 751, 'idx_b': 767})
    test_list.append({'key': '01689', 'idx_a': 900, 'idx_b': 649})
    test_list.append({'key': '00178', 'idx_a': 275, 'idx_b': 34})
    test_list.append({'key': '00190', 'idx_a': 92, 'idx_b': 107})
    test_list.append({'key': '02348', 'idx_a': 93, 'idx_b': 525})
    test_list.append({'key': '02833', 'idx_a': 2003, 'idx_b': 1412})
    test_list.append({'key': '02966', 'idx_a': 843, 'idx_b': 4})
    test_list.append({'key': '00319', 'idx_a': 1220, 'idx_b': 2700})
    test_list.append({'key': '03366', 'idx_a': 177, 'idx_b': 27})
    test_list.append({'key': '03404', 'idx_a': 361, 'idx_b': 640})
    test_list.append({'key': '00563', 'idx_a': 37, 'idx_b': 1810})
    test_list.append({'key': '00616', 'idx_a': 1840, 'idx_b': 280})
    test_list.append({'key': '00646', 'idx_a': 601, 'idx_b': 1793})
    test_list.append({'key': '00654', 'idx_a': 301, 'idx_b': 728}) #
    test_list.append({'key': '00777', 'idx_a': 451, 'idx_b': 293})
    test_list.append({'key': '00796', 'idx_a': 662, 'idx_b': 1470}) #
    test_list.append({'key': '00935', 'idx_a': 73, 'idx_b': 773})
    test_list.append({'key': '00953', 'idx_a': 512, 'idx_b': 1519})
    return test_list


def get_example_images(dataset, test_list):
    def retrieve(group, index):
        eyes = dataset.preprocess_image(group['pixels'][index, :])
        g = group['labels'][index, :2]
        h = group['labels'][index, 2:4]
        return eyes, g, h
    hdf = h5py.File(dataset.hdf_path, 'r', libver='latest', swmr=True)
    entries = []
    for item in test_list:
        key = item['key']
        group_a = hdf[key]
        group_b = group_a
        idx_a = item['idx_a']
        idx_b = item['idx_b']
        eyes_a, g_a, h_a = retrieve(group_a, idx_a)
        eyes_b, g_b, h_b = retrieve(group_b, idx_b)
        entry = {
            'key': torch.tensor(int(key), dtype=torch.int),
            'idx_a': torch.tensor(idx_a, dtype=torch.int),
            'idx_b': torch.tensor(idx_b, dtype=torch.int),
            'image_a': torch.tensor(eyes_a, dtype=torch.float),
            'gaze_a': torch.tensor(g_a, dtype=torch.float),
            'head_a': torch.tensor(h_a, dtype=torch.float),
            'image_b': torch.tensor(eyes_b, dtype=torch.float),
            'gaze_b': torch.tensor(g_b, dtype=torch.float),
            'head_b': torch.tensor(h_b, dtype=torch.float),
        }
        entries.append(entry)
    test_visualize = {}
    for k in ['key', 'idx_a', 'idx_b', 'image_a', 'gaze_a', 'head_a', 'image_b', 'gaze_b', 'head_b']:
        if k in entries[0]:
            test_visualize[k] = torch.stack([s[k] for s in entries])

    input_images = test_visualize['image_a']
    target_images = test_visualize['image_b']
    keys = test_visualize['key']
    for i in range(len(input_images)):
        name = str(keys[i].cpu().numpy())
        path = os.path.join(config.save_path, 'samples', name)
        if not os.path.exists(path):
            os.makedirs(path)
        cv2.imwrite(os.path.join(path, 'input_image.png'), recover_images(input_images[i]))
        cv2.imwrite(os.path.join(path, 'target_image.png'), recover_images(target_images[i]))
    return test_visualize


class RunningStatistics(object):
    def __init__(self):
        self.losses = OrderedDict()
    def add(self, key, value):
        if key not in self.losses:
            self.losses[key] = []
        self.losses[key].append(value)
    def means(self):
        return OrderedDict([
            (k, np.mean(v)) for k, v in self.losses.items() if len(v) > 0
        ])
    def reset(self):
        for key in self.losses.keys():
            self.losses[key] = []


def adjust_learning_rate(optimizers, decay, number_decay, base_lr):
    lr = base_lr * (decay ** number_decay)
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr


def script_init_common():
    parser = argparse.ArgumentParser(description='Train a gaze estimation model.')
    parser.add_argument('config_json', type=str, nargs='*',
                        help=('Path to config in JSON format. '
                              'Multiple configs will be parsed in the specified order.'))
    args = parser.parse_args()
    # Parse configs in order specified by user
    for json_path in args.config_json:
        config.import_json(json_path)


def save_model(network, current_step):
    models = {
        'encoder': network.encoder.state_dict(),
        'decoder': network.decoder.state_dict(),
        'discriminator': network.discriminator.state_dict(),
        'iin' : network.iin.state_dict(),
        'eye_extractor' : network.eye_extractor.state_dict(),
        'regressor' : network.regressor.state_dict()
    }
    p = os.path.join(config.save_path, "checkpoints")
    path = os.path.join(p, str(current_step) + '.pt')
    if not os.path.exists(p):
        os.makedirs(p)
    torch.save(models, path)


def load_model(network, path):
    checkpoint = torch.load(path, map_location=device)
    network.encoder.load_state_dict(checkpoint['encoder'])
    network.decoder.load_state_dict(checkpoint['decoder'])
    network.discriminator.load_state_dict(checkpoint['discriminator'])
    network.iin.load_state_dict(checkpoint['iin'])
    network.eye_extractor.load_state_dict(checkpoint['eye_extractor'])
    network.regressor.load_state_dict(checkpoint['regressor'])