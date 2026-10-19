# NPU-adapted variant of airbench optimized for time-to-96%.
# Adapted from airbench96_faster.py for Ascend 910 NPU.

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import time
from collections import deque
import uuid
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import torch_npu
from torch_npu.contrib import transfer_to_npu

# Set only by the Stage-2 short-run profiler.  It stays None in all full
# training and benchmark runs, preserving the original hot-path control flow.
PROFILE_STEP_HOOK = None

hyp = {
    'opt': {
        'train_epochs': 45.0,
        'batch_size': 1024,
        'batch_size_masked': 512,
        'lr': 9.0,
        'momentum': 0.85,
        'weight_decay': 0.012,
        'bias_scaler': 64.0,
        'label_smoothing': 0.2,
        'whiten_bias_epochs': 3,
    },
    'aug': {
        'flip': True,
        'translate': 4,
        'cutout': 12,
    },
    'proxy': {
        'widths': {
            'block1': 32,
            'block2': 64,
            'block3': 64,
        },
        'depth': 2,
        'scaling_factor': 1/9,
    },
    'net': {
        'widths': {
            'block1': 128,
            'block2': 384,
            'block3': 512,
        },
        'depth': 3,
        'scaling_factor': 1/9,
        'tta_level': 2,
    },
}

#############################################
#                DataLoader                 #
#############################################

CIFAR_MEAN = torch.tensor((0.4914, 0.4822, 0.4465))
CIFAR_STD = torch.tensor((0.2470, 0.2435, 0.2616))

def batch_flip_lr(inputs):
    flip_mask = (torch.rand(len(inputs), device=inputs.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip_mask, inputs.flip(-1), inputs)

def batch_crop(images, crop_size):
    r = (images.size(-1) - crop_size)//2
    shifts = torch.randint(-r, r+1, size=(len(images), 2), device=images.device)
    images_out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if r <= 2:
        for sy in range(-r, r+1):
            for sx in range(-r, r+1):
                mask = (shifts[:, 0] == sy) & (shifts[:, 1] == sx)
                images_out[mask] = images[mask, :, r+sy:r+sy+crop_size, r+sx:r+sx+crop_size]
    else:
        images_tmp = torch.empty((len(images), 3, crop_size, crop_size+2*r), device=images.device, dtype=images.dtype)
        for s in range(-r, r+1):
            mask = (shifts[:, 0] == s)
            images_tmp[mask] = images[mask, :, r+s:r+s+crop_size, :]
        for s in range(-r, r+1):
            mask = (shifts[:, 1] == s)
            images_out[mask] = images_tmp[mask, :, :, r+s:r+s+crop_size]
    return images_out

def make_random_square_masks(inputs, size):
    is_even = int(size % 2 == 0)
    n,c,h,w = inputs.shape
    corner_y = torch.randint(0, h-size+1, size=(n,), device=inputs.device)
    corner_x = torch.randint(0, w-size+1, size=(n,), device=inputs.device)
    corner_y_dists = torch.arange(h, device=inputs.device).view(1, 1, h, 1) - corner_y.view(-1, 1, 1, 1)
    corner_x_dists = torch.arange(w, device=inputs.device).view(1, 1, 1, w) - corner_x.view(-1, 1, 1, 1)
    mask_y = (corner_y_dists >= 0) * (corner_y_dists < size)
    mask_x = (corner_x_dists >= 0) * (corner_x_dists < size)
    final_mask = mask_y * mask_x
    return final_mask

def batch_cutout(inputs, size):
    cutout_masks = make_random_square_masks(inputs, size)
    return inputs.masked_fill(cutout_masks, 0)

def set_random_state(seed, state):
    if seed is None:
        import random
        torch.manual_seed(random.randint(0, 2**63))
    else:
        seed1 = 1000 * seed + state
        torch.manual_seed(seed1)

class InfiniteCifarLoader:
    """Construct every training input during iteration."""

    def __init__(self, path, train=True, batch_size=500, aug=None, altflip=True,
                 subset_mask=None, aug_seed=None, order_seed=None):
        path = os.environ.get('AIRBENCH_DATA_PATH', path)
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        data = torch.load(data_path, map_location='npu')
        self.images, self.labels, self.classes = data['images'], data['labels'], data['classes']
        self.images = (self.images.half() / 255).permute(0, 3, 1, 2)
        self.normalize = T.Normalize(CIFAR_MEAN, CIFAR_STD)
        self.aug = aug or {}
        for k in self.aug.keys():
            assert k in ['flip', 'translate', 'cutout'], 'Unrecognized key: %s' % k
        self.batch_size = batch_size
        self.altflip = altflip
        self.subset_mask = subset_mask if subset_mask is not None else torch.tensor([True]*len(self.images)).npu()
        self.train = train
        self.aug_seed = aug_seed
        self.order_seed = order_seed

    def __iter__(self):
        images0 = self.normalize(self.images)
        if self.aug.get('flip', False) and self.altflip:
            set_random_state(self.aug_seed, 0)
            images0 = batch_flip_lr(images0)
        pad = self.aug.get('translate', 0)
        if pad > 0:
            images0 = F.pad(images0, (pad,)*4, 'reflect')
        labels0 = self.labels
        epoch = 0
        batch_size = self.batch_size
        num_examples = self.subset_mask.sum().item()
        current_pointer = num_examples
        batch_images = torch.empty(0, 3, 32, 32, dtype=images0.dtype, device=images0.device)
        batch_labels = torch.empty(0, dtype=labels0.dtype, device=labels0.device)
        batch_indices = torch.empty(0, dtype=labels0.dtype, device=labels0.device)

        while True:
            assert len(batch_images) < batch_size
            if current_pointer >= num_examples:
                epoch += 1
                set_random_state(self.aug_seed, epoch)
                if pad > 0:
                    images1 = batch_crop(images0, 32)
                if self.aug.get('flip', False):
                    if self.altflip:
                        images1 = images1 if epoch % 2 == 0 else images1.flip(-1)
                    else:
                        images1 = batch_flip_lr(images1)
                if self.aug.get('cutout', 0) > 0:
                    images1 = batch_cutout(images1, self.aug['cutout'])
                set_random_state(self.order_seed, epoch)
                indices = (torch.randperm if self.train else torch.arange)(len(self.images), device=images0.device)
                indices_subset = indices[self.subset_mask[indices]]
                current_pointer = 0

            remaining_size = batch_size - len(batch_images)
            extra_indices = indices_subset[current_pointer:current_pointer+remaining_size]
            extra_images = images1[extra_indices]
            extra_labels = labels0[extra_indices]
            current_pointer += remaining_size
            batch_indices = torch.cat([batch_indices, extra_indices])
            batch_images = torch.cat([batch_images, extra_images])
            batch_labels = torch.cat([batch_labels, extra_labels])

            if len(batch_images) == batch_size:
                assert len(batch_images) == len(batch_labels)
                yield (batch_indices, batch_images, batch_labels)
                batch_images = torch.empty(0, 3, 32, 32, dtype=images0.dtype, device=images0.device)
                batch_labels = torch.empty(0, dtype=labels0.dtype, device=labels0.device)
                batch_indices = torch.empty(0, dtype=labels0.dtype, device=labels0.device)

############################################
#               Evaluation                 #
############################################

def infer(model, loader, tta_level=0):

    def infer_basic(inputs, net):
        return net(inputs).clone()

    def infer_mirror(inputs, net):
        return 0.5 * net(inputs) + 0.5 * net(inputs.flip(-1))

    def infer_mirror_translate(inputs, net):
        logits = infer_mirror(inputs, net)
        pad = 1
        padded_inputs = F.pad(inputs, (pad,)*4, 'reflect')
        inputs_translate_list = [
            padded_inputs[:, :, 0:32, 0:32],
            padded_inputs[:, :, 2:34, 2:34],
        ]
        logits_translate_list = [infer_mirror(inputs_translate, net)
                                 for inputs_translate in inputs_translate_list]
        logits_translate = torch.stack(logits_translate_list).mean(0)
        return 0.5 * logits + 0.5 * logits_translate

    model.eval()
    test_images = loader.normalize(loader.images)
    eval_chunk = int(os.environ.get('AIRBENCH_EVAL_CHUNK', '2000'))
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        if PROFILE_STEP_HOOK is None:
            return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(eval_chunk)])
        logits = []
        for profile_step, inputs in enumerate(test_images.split(eval_chunk), 1):
            logits.append(infer_fn(inputs, model))
            PROFILE_STEP_HOOK('eval', profile_step)
        return torch.cat(logits)

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class FirstMaxPool3(nn.Module):
    """Exact MaxPool2d(3) for a 3x3 input, including first-tie gradients."""
    def forward(self, x):
        x = torch.max(x, dim=-1).values
        return torch.max(x, dim=-1, keepdim=True).values.unsqueeze(-1)

class Mul(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
    def forward(self, x):
        return x * self.scale

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-12,
                 weight=False, bias=True):
        super().__init__(num_features, eps=eps)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class Conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='same', bias=False):
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out, depth):
        super().__init__()
        assert depth in (2, 3)
        self.depth = depth
        self.conv1 = Conv(channels_in,  channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        if depth == 3:
            self.conv3 = Conv(channels_out, channels_out)
            self.norm3 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.norm1(x)
        x = self.activ(x)
        if self.depth == 3:
            x0 = x
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.activ(x)
        if self.depth == 3:
            x = self.conv3(x)
            x = self.norm3(x)
            x = x + x0
            x = self.activ(x)
        return x

#############################################
#            Network Definition             #
#############################################

def make_net(hyp):
    widths = hyp['widths']
    scaling_factor = hyp['scaling_factor']
    depth = hyp['depth']
    whiten_kernel_size = 2
    whiten_width = 2 * 3 * whiten_kernel_size**2
    net = nn.Sequential(
        Conv(3, whiten_width, whiten_kernel_size, padding=0, bias=True),
        nn.GELU(),
        ConvGroup(whiten_width,     widths['block1'], depth),
        ConvGroup(widths['block1'], widths['block2'], depth),
        ConvGroup(widths['block2'], widths['block3'], depth),
        FirstMaxPool3() if os.environ.get('AIRBENCH_FINAL_REDUCE_POOL', '0') == '1' else nn.MaxPool2d(3),
        Flatten(),
        nn.Linear(widths['block3'], 10, bias=False),
        Mul(scaling_factor),
    )
    net[0].weight.requires_grad = False
    net = net.half().npu()
    for mod in net.modules():
        if isinstance(mod, BatchNorm):
            mod.float()
    return net

def reinit_net(model):
    for m in model.modules():
        if type(m) in (Conv, BatchNorm, nn.Linear):
            m.reset_parameters()

#############################################
#       Whitening Conv Initialization       #
#############################################

def get_patches(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3,w,1).transpose(1,3).reshape(-1,c,h,w).float()

def get_whitening_parameters(patches):
    n,c,h,w = patches.shape
    patches_flat = patches.view(n, -1)
    est_patch_covariance = (patches_flat.T @ patches_flat) / n
    eigenvalues, eigenvectors = torch.linalg.eigh(est_patch_covariance, UPLO='U')
    return eigenvalues.flip(0).view(-1, 1, 1, 1), eigenvectors.T.reshape(c*h*w,c,h,w).flip(0)

def init_whitening_conv(layer, train_set, eps=5e-4):
    patches = get_patches(train_set, patch_shape=layer.weight.data.shape[2:])
    eigenvalues, eigenvectors = get_whitening_parameters(patches)
    eigenvectors_scaled = eigenvectors / torch.sqrt(eigenvalues + eps)
    layer.weight.data[:] = torch.cat((eigenvectors_scaled, -eigenvectors_scaled))

############################################
#                Lookahead                 #
############################################

class LookaheadState:
    def __init__(self, net):
        self.net_ema = {k: v.clone() for k, v in net.state_dict().items()}

    def update(self, net, decay):
        for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
            if net_param.dtype in (torch.half, torch.float):
                ema_param.lerp_(net_param, 1-decay)
                net_param.copy_(ema_param)

############################################
#                 Logging                  #
############################################

def print_columns(columns_list, is_head=False, is_final_entry=False):
    print_string = ''
    for col in columns_list:
        print_string += '|  %s  ' % col
    print_string += '|'
    if is_head:
        print('-'*len(print_string))
    print(print_string)
    if is_head or is_final_entry:
        print('-'*len(print_string))

logging_columns_list = ['run   ', 'epoch', 'train_loss', 'train_acc', 'val_acc', 'tta_val_acc', 'total_time_seconds']
def print_training_details(variables, is_final_entry):
    formatted = []
    for col in logging_columns_list:
        var = variables.get(col.strip(), None)
        if type(var) in (int, str):
            res = str(var)
        elif type(var) is float:
            res = '{:0.4f}'.format(var)
        else:
            assert var is None
            res = ''
        formatted.append(res.rjust(len(col)))
    print_columns(formatted, is_final_entry=is_final_entry)

############################################
#                Training                  #
############################################

def train_proxy(hyp, model, data_seed):

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    train_loader = InfiniteCifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'],
                                       aug_seed=data_seed, order_seed=data_seed)
    steps_per_epoch = len(train_loader.images) // batch_size
    total_train_steps = ceil(steps_per_epoch * epochs)

    set_random_state(None, 0)
    reinit_net(model)
    print('Proxy parameters:', sum(p.numel() for p in model.parameters()))
    current_steps = 0

    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.1)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (total_train_steps - step) / warmdown_steps
            return frac
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)

    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images)

    masks = []

    for indices, inputs, labels in train_loader:

        if current_steps % steps_per_epoch == 0:
            epoch = current_steps // steps_per_epoch
            model.train()

        if current_steps % 4 == 0:
            outputs = model(inputs)
            loss1 = loss_fn(outputs, labels)
            mask = torch.zeros(len(inputs)).npu().bool()
            mask[loss1.argsort()[-hyp['opt']['batch_size_masked']:]] = True
            masks.append(mask)
            loss = (loss1 * mask.float()).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss1 = loss_fn(outputs, labels)
                mask = torch.zeros(len(inputs)).npu().bool()
                mask[loss1.argsort()[-hyp['opt']['batch_size_masked']:]] = True
                masks.append(mask)

        scheduler.step()

        current_steps += 1
        if PROFILE_STEP_HOOK is not None:
            PROFILE_STEP_HOOK('proxy', current_steps)
        if current_steps == total_train_steps:
            break

    return masks


def make_lazy_proxy_step(hyp, model, data_seed, train_loader):
    """Build an on-demand proxy step that reuses the main loader's batch."""
    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']
    steps_per_epoch = len(train_loader.images) // batch_size
    total_train_steps = ceil(steps_per_epoch * epochs)

    loss_fn = nn.CrossEntropyLoss(
        label_smoothing=hyp['opt']['label_smoothing'], reduction='none'
    )
    set_random_state(None, 0)
    reinit_net(model)
    print('Proxy parameters:', sum(p.numel() for p in model.parameters()))

    norm_biases = [p for k, p in model.named_parameters() if 'norm' in k and p.requires_grad]
    other_params = [p for k, p in model.named_parameters() if 'norm' not in k and p.requires_grad]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.1)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        frac = (total_train_steps - step) / warmdown_steps
        return frac

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr)
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model[0], train_images)
    current_steps = 0

    def step(inputs, labels):
        nonlocal current_steps
        if current_steps % steps_per_epoch == 0:
            model.train()

        if current_steps % 4 == 0:
            outputs = model(inputs)
            loss1 = loss_fn(outputs, labels)
            mask = torch.zeros(len(inputs)).npu().bool()
            mask[loss1.argsort()[-hyp['opt']['batch_size_masked']:]] = True
            loss = (loss1 * mask.float()).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss1 = loss_fn(outputs, labels)
                mask = torch.zeros(len(inputs)).npu().bool()
                mask[loss1.argsort()[-hyp['opt']['batch_size_masked']:]] = True

        scheduler.step()
        current_steps += 1
        if PROFILE_STEP_HOOK is not None:
            PROFILE_STEP_HOOK('proxy', current_steps)
        return mask

    return step


def prefetch_proxy_batches(train_loader, proxy_step, prefetch_steps):
    """Prefetch a bounded mask window while retaining the exact input batches."""
    loader = iter(train_loader)
    while True:
        t0 = time.time()
        window = []
        for _ in range(prefetch_steps):
            indices, inputs, labels = next(loader)
            mask = proxy_step(inputs, labels)
            window.append((indices, inputs, labels, mask))
        torch.npu.synchronize()
        prefetch_seconds = time.time() - t0
        for offset, (indices, inputs, labels, mask) in enumerate(window):
            yield (
                indices,
                inputs,
                labels,
                mask,
                prefetch_seconds if offset == 0 else 0.0,
            )

_PROXY_STREAM = None


def stream_proxy_batches(train_loader, proxy_step, total_steps, lookahead_steps):
    """Run bounded proxy lookahead on an independent NPU stream.

    The consumer stream waits only for the mask it is about to use.  Epoch-end
    device synchronization in ``main`` accounts for all producer work in the
    official timer without turning each lookahead window into a serial phase.
    """
    global _PROXY_STREAM
    loader = iter(train_loader)
    consumer_stream = torch.npu.current_stream()
    if _PROXY_STREAM is None:
        proxy_priority = int(os.environ.get('AIRBENCH_PROXY_STREAM_PRIORITY', '0'))
        _PROXY_STREAM = torch.npu.Stream(priority=proxy_priority)
    producer_stream = _PROXY_STREAM
    queued = deque()
    launched = 0

    def launch_one():
        nonlocal launched
        with torch.npu.stream(producer_stream):
            indices, inputs, labels = next(loader)
            mask = proxy_step(inputs, labels)
            ready = producer_stream.record_event()
            for tensor in (indices, inputs, labels, mask):
                tensor.record_stream(producer_stream)
        queued.append((indices, inputs, labels, mask, ready))
        launched += 1

    for _ in range(min(lookahead_steps, total_steps)):
        launch_one()

    while queued:
        indices, inputs, labels, mask, ready = queued.popleft()
        consumer_stream.wait_event(ready)
        for tensor in (indices, inputs, labels, mask):
            tensor.record_stream(consumer_stream)
        yield indices, inputs, labels, mask, 0.0
        if launched < total_steps:
            launch_one()

def main(run, hyp, model_proxy, model_trainbias, model_freezebias):

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    set_random_state(None, 0)
    import random
    data_seed = random.randint(0, 2**50)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    test_loader = InfiniteCifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = InfiniteCifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'],
                                       aug_seed=data_seed, order_seed=data_seed)
    steps_per_epoch = len(train_loader.images) // batch_size
    total_train_steps = ceil(steps_per_epoch * epochs)

    set_random_state(None, 0)
    reinit_net(model_trainbias)
    print('Main model parameters:', sum(p.numel() for p in model_trainbias.parameters()))
    current_steps = 0

    norm_biases = [p for k, p in model_trainbias.named_parameters() if 'norm' in k]
    other_params = [p for k, p in model_trainbias.named_parameters() if 'norm' not in k]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer_trainbias = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    norm_biases = [p for k, p in model_freezebias.named_parameters() if 'norm' in k]
    other_params = [p for k, p in model_freezebias.named_parameters() if 'norm' not in k]
    param_configs = [dict(params=norm_biases, lr=lr_biases, weight_decay=wd/lr_biases),
                     dict(params=other_params, lr=lr, weight_decay=wd/lr)]
    optimizer_freezebias = torch.optim.SGD(param_configs, momentum=momentum, nesterov=True)

    def get_lr(step):
        warmup_steps = int(total_train_steps * 0.1)
        warmdown_steps = total_train_steps - warmup_steps
        if step < warmup_steps:
            frac = step / warmup_steps
            return 0.2 * (1 - frac) + 1.0 * frac
        else:
            frac = (total_train_steps - step) / warmdown_steps
            return frac
    scheduler_trainbias = torch.optim.lr_scheduler.LambdaLR(optimizer_trainbias, get_lr)
    scheduler_freezebias = torch.optim.lr_scheduler.LambdaLR(optimizer_freezebias, get_lr)

    alpha_schedule = 0.95**5 * (torch.arange(total_train_steps+1) / total_train_steps)**3
    lookahead_state = LookaheadState(model_trainbias)

    import time
    total_time_seconds = 0.0

    t0 = time.time()
    train_images = train_loader.normalize(train_loader.images[:5000])
    init_whitening_conv(model_trainbias[0], train_images)
    torch.npu.synchronize()
    total_time_seconds += time.time() - t0

    print('Training small proxy...')
    t0 = time.time()
    lazy_proxy = os.environ.get('AIRBENCH_LAZY_PROXY_MASKS', '0') == '1'
    stream_proxy = os.environ.get('AIRBENCH_STREAM_PROXY_MASKS', '0') == '1'
    if lazy_proxy or stream_proxy:
        proxy_step = make_lazy_proxy_step(hyp, model_proxy, data_seed, train_loader)
        masks = None
        if stream_proxy:
            lookahead_steps = int(os.environ.get('AIRBENCH_PROXY_LOOKAHEAD_STEPS', '4'))
            assert 1 <= lookahead_steps <= total_train_steps
            train_batches = stream_proxy_batches(
                train_loader, proxy_step, total_train_steps, lookahead_steps
            )
        else:
            prefetch_steps = int(os.environ.get('AIRBENCH_PROXY_PREFETCH_STEPS', steps_per_epoch))
            assert 1 <= prefetch_steps <= total_train_steps
            train_batches = prefetch_proxy_batches(train_loader, proxy_step, prefetch_steps)
    else:
        masks = iter(train_proxy(hyp, model_proxy, data_seed))
        proxy_step = None
        train_batches = (
            (indices, inputs, labels, None, 0.0)
            for indices, inputs, labels in train_loader
        )
    torch.npu.synchronize()
    total_time_seconds += time.time() - t0

    for indices, inputs, labels, prefetched_mask, prefetch_seconds in train_batches:

        total_time_seconds += prefetch_seconds

        if current_steps == 0:
            model = model_trainbias
            optimizer = optimizer_trainbias
            scheduler = scheduler_trainbias
        elif epoch == hyp['opt']['whiten_bias_epochs'] * steps_per_epoch:
            model = model_freezebias
            optimizer = optimizer_freezebias
            scheduler = scheduler_freezebias
            model.load_state_dict(model_trainbias.state_dict())
            optimizer.load_state_dict(optimizer_trainbias.state_dict())
            scheduler.load_state_dict(scheduler_trainbias.state_dict())

        ####################
        #     Training     #
        ####################

        if current_steps % steps_per_epoch == 0:
            epoch = current_steps // steps_per_epoch
            t0 = time.time()
            model.train()

        mask = prefetched_mask if (lazy_proxy or stream_proxy) else next(masks)
        inputs = inputs[mask]
        labels = labels[mask]
        outputs = model(inputs)
        loss = loss_fn(outputs, labels).sum()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        current_steps += 1
        if current_steps % 5 == 0:
            lookahead_state.update(model, decay=alpha_schedule[current_steps].item())
        if current_steps == total_train_steps:
            if lookahead_state is not None:
                lookahead_state.update(model, decay=1.0)

        if PROFILE_STEP_HOOK is not None:
            PROFILE_STEP_HOOK('main', current_steps)

        if (current_steps % steps_per_epoch == 0) or (current_steps == total_train_steps):
            torch.npu.synchronize()
            total_time_seconds += time.time() - t0

            ####################
            #    Evaluation    #
            ####################

            train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
            train_loss = loss.item() / batch_size
            val_acc = evaluate(model, test_loader, tta_level=0)
            print_training_details(locals(), is_final_entry=False)
            run = None

        if current_steps == total_train_steps:
            break

    ####################
    #  TTA Evaluation  #
    ####################

    t0 = time.time()
    tta_val_acc = evaluate(model, test_loader, tta_level=hyp['net']['tta_level'])
    torch.npu.synchronize()
    total_time_seconds += time.time() - t0

    epoch = 'eval'
    print_training_details(locals(), is_final_entry=True)

    return tta_val_acc, total_time_seconds

if __name__ == "__main__":
    with open(sys.argv[0]) as f:
        code = f.read()

    model_proxy = make_net(hyp['proxy'])
    model_proxy[0].bias.requires_grad = False
    model_trainbias = make_net(hyp['net'])
    model_freezebias = make_net(hyp['net'])
    model_freezebias[0].bias.requires_grad = False

    print_columns(logging_columns_list, is_head=True)
    results = [main(run, hyp, model_proxy, model_trainbias, model_freezebias)
               for run in range(int(os.environ.get('AIRBENCH_RUNS', '200')))]
    accs = torch.tensor([r[0] for r in results])
    times = torch.tensor([r[1] for r in results])
    print('Mean: %.4f    Std: %.4f' % (accs.mean(), accs.std()))
    print('Time per run - Mean: %.2fs    Std: %.2fs    Min: %.2fs    Max: %.2fs' %
          (times.mean(), times.std(), times.min(), times.max()))

    log = {'code': code, 'accs': accs, 'times': times}
    log_dir = os.path.join('logs', str(uuid.uuid4()))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'log.pt')
    print(os.path.abspath(log_path))
    torch.save(log, os.path.join(log_dir, 'log.pt'))
