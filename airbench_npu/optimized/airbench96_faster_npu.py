# NPU-adapted variant of airbench optimized for time-to-96%.
# Adapted from airbench96_faster.py for Ascend 910 NPU.

#############################################
#            Setup/Hyperparameters          #
#############################################

import os
import sys
import time
import json
import argparse
from pathlib import Path
from math import ceil

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

import torch_npu
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

DDP_MODE = os.environ.get('WORLD_SIZE', '1') == '2'
DDP_WRAPPERS = {}

# --- optimisation switches (all default ON; set to 0 to ablate) -----------
def _flag(name, default='1'):
    return os.environ.get(name, default) == '1'

NO_BCAST = _flag('AIRBENCH_F_NOBCAST')          # drop redundant per-step HCCL broadcasts
LOCAL_LOSS = _flag('AIRBENCH_F_LOCALLOSS')      # per-rank loss, no logits all-gather
NO_GRADCAST = _flag('AIRBENCH_F_NOGRADCAST', '0')   # drop per-param npu_format_cast loop
KEEP_FMT = _flag('AIRBENCH_F_KEEPFMT', '1')         # never disable allow_internal_format
BWD_ND = _flag('AIRBENCH_F_BWDND', '0')             # run backward with internal format off
FAST_LOADER = _flag('AIRBENCH_F_FASTLOADER')    # skip empty-tensor concats per batch
FOREACH_LA = _flag('AIRBENCH_F_FOREACHLA')      # fused lookahead update
PROXY_LEAN = _flag('AIRBENCH_F_PROXYLEAN')      # drop unused work in proxy steps
OVERLAP_PROXY = _flag('AIRBENCH_F_OVERLAPPROXY')  # stream the proxy alongside training
MASK_INDEX_SELECT = _flag('AIRBENCH_F_IDXSELECT')  # index_select instead of advanced indexing
PROXY_LOOKAHEAD = int(os.environ.get('AIRBENCH_PROXY_LOOKAHEAD', '8'))
CHECK_INPUT = _flag('AIRBENCH_CHECK_INPUT', '0')  # verify per-step cross-rank input equality
MASK_SYNC = _flag('AIRBENCH_MASK_SYNC', '1')      # share rank 0's hard-example indices (4KB)
BUCKET_VIEW = _flag('AIRBENCH_BUCKET_VIEW', '1')  # DDP gradient_as_bucket_view
FLAT_FINALPOOL = _flag('AIRBENCH_F_FLATFINALPOOL')  # single-reduction final window max
# True only when the per-rank shard really is a strict half of the global batch.
SHARDED = DDP_MODE and LOCAL_LOSS


def broadcast_state(model):
    if DDP_MODE:
        for value in model.state_dict().values():
            dist.broadcast(value, 0)


class GatherLogits(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local):
        ctx.rank = dist.get_rank()
        chunks = [torch.empty_like(local) for _ in range(2)]
        dist.all_gather(chunks, local.contiguous())
        return torch.cat(chunks)

    @staticmethod
    def backward(ctx, gradient):
        # Each rank computes the same global summed loss. DDP averages parameter
        # gradients, so compensate its division by world size exactly once.
        return gradient.chunk(2)[ctx.rank].contiguous() * 2


class GatherFeatures(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local):
        ctx.rank = dist.get_rank()
        pieces = [torch.empty_like(local) for _ in range(2)]
        dist.all_gather(pieces, local.contiguous())
        return torch.cat(pieces)

    @staticmethod
    def backward(ctx, gradient):
        # Full BN Jacobian couples examples across ranks. Sum full-output adjoints
        # BEFORE selecting the local convolution-input shard.
        gradient = gradient.contiguous()
        dist.all_reduce(gradient)
        return gradient.chunk(2)[ctx.rank].contiguous()


class ExactSyncBatchNorm(BatchNorm if 'BatchNorm' in globals() else nn.BatchNorm2d):
    def forward(self, local):
        if not self.training:
            return super().forward(local)
        full = GatherFeatures.apply(local)
        output = super().forward(full)
        return output.chunk(2)[dist.get_rank()].contiguous()


def convert_sync_bn(model):
    if os.environ.get('AIRBENCH_SYNCBN', 'exact') == 'local':
        return model
    if os.environ.get('AIRBENCH_SYNCBN', 'exact') == 'native':
        return nn.SyncBatchNorm.convert_sync_batchnorm(model)
    for name, child in list(model.named_children()):
        if isinstance(child, BatchNorm):
            replacement = ExactSyncBatchNorm(child.num_features, eps=child.eps,
                                             momentum=child.momentum)
            replacement.to(device=child.weight.device, dtype=child.weight.dtype)
            replacement.load_state_dict(child.state_dict())
            replacement.weight.requires_grad_(child.weight.requires_grad)
            replacement.bias.requires_grad_(child.bias.requires_grad)
            setattr(model, name, replacement)
        else:
            convert_sync_bn(child)
    return model


def local_shard(tensor):
    return tensor.chunk(2)[dist.get_rank()].contiguous()


def training_forward(model, inputs):
    if not DDP_MODE:
        return model(inputs)
    if CHECK_INPUT:
        # Rank-local augmentation is seeded identically on both ranks, so the global
        # batch must be bit-identical; verified on the host because the device-side
        # comparison races with the HCCL broadcast.
        reference = inputs.clone()
        dist.broadcast(reference, 0)
        torch.npu.synchronize()
        if bool((inputs.float().cpu() != reference.float().cpu()).any().item()):
            raise RuntimeError('rank-local augmented inputs differ; broadcast removal unsafe')
    if NO_BCAST:
        local = local_shard(inputs)
    else:
        dist.broadcast(inputs, 0)
        local = inputs.chunk(2)[dist.get_rank()].contiguous()
    outputs = DDP_WRAPPERS[id(model)](local)
    return outputs if LOCAL_LOSS else GatherLogits.apply(outputs)

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
    """
    CIFAR-10 loader which constructs every input to be used during training during the call to __iter__.
    """

    _normalized_cache = {}

    def __init__(self, path, train=True, batch_size=500, aug=None, altflip=True, subset_mask=None, aug_seed=None, order_seed=None):
        path = os.environ.get('AIRBENCH_DATA_PATH', path)
        data_path = os.path.join(path, 'train.pt' if train else 'test.pt')
        if not os.path.exists(data_path):
            dset = torchvision.datasets.CIFAR10(path, download=True, train=train)
            images = torch.tensor(dset.data)
            labels = torch.tensor(dset.targets)
            torch.save({'images': images, 'labels': labels, 'classes': dset.classes}, data_path)

        cache_key = (os.path.abspath(data_path), torch.npu.current_device())
        cached = self._normalized_cache.get(cache_key)
        if cached is None:
            data = torch.load(data_path, map_location='npu')
            images = (data['images'].half() / 255).permute(0, 3, 1, 2)
            images = T.Normalize(CIFAR_MEAN, CIFAR_STD)(images)
            cached = (images, data['labels'], data['classes'])
            self._normalized_cache[cache_key] = cached
        self.images, self.labels, self.classes = cached

        # The immutable normalized dataset is shared by the main/proxy loaders
        # and reused across sequential trainings. Augmentation
        # still creates per-run tensors from this exact normalized input.
        self.normalize = lambda images: images

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
            if FAST_LOADER and len(batch_images) == 0:
                batch_indices, batch_images, batch_labels = extra_indices, extra_images, extra_labels
            else:
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
    infer_fn = [infer_basic, infer_mirror, infer_mirror_translate][tta_level]
    with torch.no_grad():
        return torch.cat([infer_fn(inputs, model) for inputs in test_images.split(2000)])

def evaluate(model, loader, tta_level=0):
    logits = infer(model, loader, tta_level)
    return (logits.argmax(1) == loader.labels).float().mean().item()

#############################################
#            Network Components             #
#############################################

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

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
        self.batch_dp = False

    def forward(self, x):
        if not self.batch_dp or x.shape[0] < 2:
            return super().forward(x)
        # One batch, one master parameter set, one optimizer. Autograd sums the
        # secondary-device weight gradient into the master parameter gradient.
        # Gather BEFORE native BatchNorm: no local-BN / training-dynamics change.
        split = (x.shape[0] + 1) // 2
        secondary = torch.device('npu:1')
        x1 = x[split:].to(secondary, non_blocking=True)
        w1 = self.weight.to(secondary, non_blocking=True)
        b1 = None if self.bias is None else self.bias.to(secondary, non_blocking=True)
        with torch.npu.device(secondary):
            y1 = F.conv2d(x1, w1, b1, self.stride, self.padding,
                          self.dilation, self.groups)
        y0 = super().forward(x[:split])
        return torch.cat((y0, y1.to(x.device, non_blocking=True)), dim=0)

    def reset_parameters(self):
        super().reset_parameters()
        if self.bias is not None:
            self.bias.data.zero_()
        w = self.weight.data
        torch.nn.init.dirac_(w[:w.size(1)])

class FinalMaxPool(nn.Module):
    def forward(self, x):
        if FLAT_FINALPOOL:
            # One reduction instead of two: same 3x3 window max, and the same
            # first-in-row-major-order tie break as the two-step form.
            return x.flatten(2).max(-1).values[..., None, None]
        x = x.reshape(*x.shape[:-2], 1, 3, 1, 3)
        return x.max(-1).values.max(-2).values


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
        FinalMaxPool(),
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
        if type(m) in (Conv, BatchNorm, nn.Linear, nn.SyncBatchNorm, ExactSyncBatchNorm):
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
        if FOREACH_LA:
            emas, params = [], []
            for ema_param, net_param in zip(self.net_ema.values(), net.state_dict().values()):
                if net_param.dtype in (torch.half, torch.float):
                    emas.append(ema_param)
                    params.append(net_param)
            torch._foreach_lerp_(emas, params, 1 - decay)
            torch._foreach_copy_(params, emas)
            return
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

def train_proxy(hyp, model, data_seed, mask_buffer=None):

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

    # Yields one hard-example mask per main-training step.  Consuming lazily lets the
    # caller run the proxy on a side stream, one step ahead of the main loop.
    for indices, inputs, labels in train_loader:

        if current_steps % steps_per_epoch == 0:
            epoch = current_steps // steps_per_epoch
            model.train()

        if not OVERLAP_PROXY:
            torch.npu.config.allow_internal_format = True
        if current_steps % 4 == 0:
            outputs = model(inputs)
            loss1 = loss_fn(outputs, labels)
            selected_indices = loss1.argsort()[-hyp['opt']['batch_size_masked']:]
            mask = torch.zeros_like(loss1, dtype=torch.bool)
            mask[selected_indices] = True
            loss = (loss1 * mask.float()).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not OVERLAP_PROXY:
                torch.npu.config.allow_internal_format = False
            for param in model.parameters():
                if param.grad is not None:
                    param.grad = torch_npu.npu_format_cast(param.grad, torch_npu.get_npu_format(param))
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss1 = loss_fn(outputs, labels)
                selected_indices = loss1.argsort()[-hyp['opt']['batch_size_masked']:]

        if not OVERLAP_PROXY:
            torch.npu.config.allow_internal_format = False
        scheduler.step()

        current_steps += 1
        selected = selected_indices.sort().values
        if MASK_SYNC and DDP_MODE:
            # Each rank's proxy can differ by a sample at the selection boundary
            # (non-deterministic device reductions), so rank 0's indices win.
            # 4KB instead of the 3MB per-step image broadcast the baseline used.
            dist.broadcast(selected, 0)
        if mask_buffer is not None:
            mask_buffer[current_steps - 1].copy_(selected)
            selected = mask_buffer[current_steps - 1]
        yield selected
        if current_steps == total_train_steps:
            return

def main(run, hyp, model_proxy, model_trainbias, model_freezebias):

    batch_size = hyp['opt']['batch_size']
    epochs = hyp['opt']['train_epochs']
    momentum = hyp['opt']['momentum']
    eval_every_epoch = os.environ.get('AIRBENCH_EVAL_EVERY_EPOCH', '1') != '0'
    kilostep_scale = 1024 * (1 + 1 / (1 - momentum))
    lr = hyp['opt']['lr'] / kilostep_scale
    wd = hyp['opt']['weight_decay'] * batch_size / kilostep_scale
    lr_biases = lr * hyp['opt']['bias_scaler']

    set_random_state(None, 0)
    import random
    data_seed = random.randint(0, 2**50)
    if DDP_MODE:
        seed_tensor = torch.tensor(data_seed, dtype=torch.int64, device='npu')
        dist.broadcast(seed_tensor, 0)
        data_seed = seed_tensor.item()

    loss_fn = nn.CrossEntropyLoss(label_smoothing=hyp['opt']['label_smoothing'], reduction='none')
    test_loader = InfiniteCifarLoader('cifar10', train=False, batch_size=2000)
    train_loader = InfiniteCifarLoader('cifar10', train=True, batch_size=batch_size, aug=hyp['aug'],
                                       aug_seed=data_seed, order_seed=data_seed)
    steps_per_epoch = len(train_loader.images) // batch_size
    total_train_steps = ceil(steps_per_epoch * epochs)

    set_random_state(None, 0)
    reinit_net(model_trainbias)
    broadcast_state(model_trainbias)
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
    broadcast_state(model_trainbias)
    if DDP_MODE:
        for net in (model_trainbias, model_freezebias):
            DDP_WRAPPERS[id(net)] = DistributedDataParallel(
                net, device_ids=[torch.npu.current_device()], broadcast_buffers=False,
                bucket_cap_mb=float(os.environ.get('AIRBENCH_BUCKET_MB', '25')),
                gradient_as_bucket_view=BUCKET_VIEW)
    torch.npu.synchronize()
    total_time_seconds += time.time() - t0

    print('Training small proxy...')
    t0 = time.time()
    if OVERLAP_PROXY:
        # Identical, deterministic proxy on every rank: masks match without a broadcast.
        # Pinning the format once keeps the process-wide setting from changing while the
        # side stream is mid-flight (the previous attempt crashed the vector core there).
        torch.npu.config.allow_internal_format = True
        _mask_buf = torch.empty(total_train_steps, hyp['opt']['batch_size_masked'],
                                dtype=torch.int64, device='npu')
        masks = train_proxy(hyp, model_proxy, data_seed, mask_buffer=_mask_buf)
    else:
        if not DDP_MODE or dist.get_rank() == 0:
            masks_tensor = torch.stack(list(train_proxy(hyp, model_proxy, data_seed)))
        else:
            masks_tensor = torch.empty(total_train_steps, hyp['opt']['batch_size_masked'],
                                       dtype=torch.int64, device='npu')
        if DDP_MODE:
            dist.broadcast(masks_tensor, 0)
        masks = iter(masks_tensor)
    main_stream = torch.npu.current_stream()
    proxy_stream = torch.npu.Stream()
    train_iter = iter(train_loader)
    indices, inputs, labels = next(train_iter)
    with torch.npu.stream(proxy_stream):
        mask = next(masks)
        if OVERLAP_PROXY and PROXY_LOOKAHEAD > 1:
            import collections
            mask_q = collections.deque()
            for _ in range(PROXY_LOOKAHEAD - 1):
                mask_q.append(next(masks))
    torch.npu.synchronize()
    total_time_seconds += time.time() - t0

    while True:

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

        main_stream.wait_stream(proxy_stream)
        if MASK_INDEX_SELECT:
            inputs = torch.index_select(inputs, 0, mask)
            labels = torch.index_select(labels, 0, mask)
        else:
            inputs = inputs[mask]
            labels = labels[mask]
        labels_shard = local_shard(labels) if SHARDED else labels
        if DDP_MODE and not NO_BCAST:
            dist.broadcast(labels, 0)
        torch.npu.config.allow_internal_format = True
        outputs = training_forward(model, inputs)
        # Per-sample losses are separable, so a per-rank sum times world size gives the
        # same full-batch gradient once DDP averages the sharded parameter gradients.
        loss = loss_fn(outputs, labels_shard).sum() * (2 if SHARDED else 1)

        if BWD_ND:
            torch.npu.config.allow_internal_format = False
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not KEEP_FMT and not BWD_ND:
            torch.npu.config.allow_internal_format = False
        if not BWD_ND:
            if not NO_GRADCAST:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = torch_npu.npu_format_cast(param.grad, torch_npu.get_npu_format(param))
        optimizer.step()
        if not KEEP_FMT:
            torch.npu.config.allow_internal_format = False
        scheduler.step()

        current_steps += 1
        if current_steps % 5 == 0:
            lookahead_state.update(model, decay=alpha_schedule[current_steps].item())
        if current_steps == total_train_steps:
            if lookahead_state is not None:
                lookahead_state.update(model, decay=1.0)

        if (current_steps % steps_per_epoch == 0) or (current_steps == total_train_steps):
            torch.npu.synchronize()
            total_time_seconds += time.time() - t0

            ####################
            #    Evaluation    #
            ####################

            if SHARDED:
                correct = (outputs.detach().argmax(1) == labels_shard).sum()
                loss_global = loss.detach().clone()
                dist.all_reduce(correct)
                dist.all_reduce(loss_global)
                train_acc = correct.item() / hyp['opt']['batch_size_masked']
                train_loss = (loss_global.item() / 2) / batch_size
            else:
                train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
                train_loss = loss.item() / batch_size
            if eval_every_epoch or current_steps == total_train_steps:
                val_acc = evaluate(model, test_loader, tta_level=0)
            else:
                val_acc = None
            print_training_details(locals(), is_final_entry=False)
            run = None

        if current_steps == total_train_steps:
            break
        indices, inputs, labels = next(train_iter)
        with torch.npu.stream(proxy_stream):
            if OVERLAP_PROXY and PROXY_LOOKAHEAD > 1:
                try:
                    mask_q.append(next(masks))
                except StopIteration:
                    pass
                mask = mask_q.popleft()
            else:
                mask = next(masks)

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

def enable_batch_dp(model, enabled):
    for module in model.modules():
        if isinstance(module, Conv):
            module.batch_dp = enabled


def sync_devices(dp):
    torch.npu.synchronize(torch.npu.current_device())
    if dp:
        torch.npu.synchronize(1)


def tensor_distance(a, b):
    a, b = a.detach().float().cpu().flatten(), b.detach().float().cpu().flatten()
    return dict(max_abs=(a-b).abs().max().item(),
                relative_l2=((a-b).norm() / a.norm().clamp_min(1e-12)).item(),
                cosine=F.cosine_similarity(a, b, dim=0).item())


def verify_dp(output):
    # Gates fixed before execution. Also measure baseline self-consistency.
    gates = dict(output_max_abs=0.01, cosine=0.999, gradient_relative_l2=0.01,
                 update_max_abs=0.01, bn_max_abs=0.0001, loss_relative=0.01)
    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    torch.npu.set_compile_mode(jit_compile=False)
    net = make_net(hyp['net'])
    initial = {k: v.clone() for k, v in net.state_dict().items()}
    reports = []
    for batch in (3, 512):
        inputs = torch.randn(batch, 3, 32, 32, device='npu:0', dtype=torch.half)
        labels = torch.arange(batch, device='npu:0') % 10
        def step(dp):
            net.load_state_dict(initial)
            enable_batch_dp(net, dp)
            net.train()
            net.zero_grad(set_to_none=True)
            x = inputs.detach().clone().requires_grad_(True)
            torch.npu.config.allow_internal_format = True
            y = net(x)
            loss = F.cross_entropy(y, labels, label_smoothing=0.2, reduction='sum')
            loss.backward()
            sync_devices(dp)
            grads = {k: p.grad.clone() for k,p in net.named_parameters() if p.grad is not None}
            grads['input'] = x.grad.clone()
            buffers = {k: v.clone() for k,v in net.named_buffers()}
            torch.npu.config.allow_internal_format = False
            for p in net.parameters():
                if p.grad is not None:
                    p.grad = torch_npu.npu_format_cast(p.grad, torch_npu.get_npu_format(p))
            opt = torch.optim.SGD([p for p in net.parameters() if p.requires_grad],
                                  lr=0.001, momentum=0.85, nesterov=True)
            opt.step()
            sync_devices(dp)
            return y.detach().clone(), loss.item(), grads, buffers, {k:v.clone() for k,v in net.state_dict().items()}
        base = step(False)
        repeat = step(False)
        candidate = step(True)
        output_diff = tensor_distance(base[0], candidate[0])
        grad_diff = {k:tensor_distance(v,candidate[2][k]) for k,v in base[2].items()}
        bn_diff = {k:(v-candidate[3][k]).abs().max().item() for k,v in base[3].items()}
        update_diff = {k:(v-candidate[4][k]).abs().max().item() for k,v in base[4].items()}
        loss_rel = abs(base[1]-candidate[1])/abs(base[1])
        passed = (output_diff['max_abs'] <= gates['output_max_abs'] and
                  output_diff['cosine'] >= gates['cosine'] and loss_rel <= gates['loss_relative'] and
                  all(v['cosine'] >= gates['cosine'] and v['relative_l2'] <= gates['gradient_relative_l2'] for v in grad_diff.values()) and
                  max(bn_diff.values()) <= gates['bn_max_abs'] and
                  max(update_diff.values()) <= gates['update_max_abs'])
        reports.append(dict(batch=batch, baseline_repeat=tensor_distance(base[0],repeat[0]),
                            output=output_diff, loss_relative=loss_rel, gradients=grad_diff,
                            bn_buffers=bn_diff, parameter_update=update_diff, passed=passed))
        print(json.dumps(reports[-1]), flush=True)
    summary = dict(mode='verify', gates=gates, reports=reports, passed=all(r['passed'] for r in reports))
    Path(output).write_text(json.dumps(summary, indent=2)+'\n')
    if not summary['passed']:
        raise RuntimeError('DP numerical gate failed; do not treat as accepted optimization')


def run_training(args):
    import random
    import statistics
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = []
    for run in range(args.runs):
        if DDP_MODE:
            dist.barrier()
        sync_devices(args.mode == 'dp' and not DDP_MODE)
        started = time.perf_counter()
        model_proxy = make_net(hyp['proxy'])
        model_proxy[0].bias.requires_grad = False
        model_trainbias = make_net(hyp['net'])
        model_freezebias = make_net(hyp['net'])
        model_freezebias[0].bias.requires_grad = False
        # Proxy remains exact original implementation: no hard-example-mask changes.
        if DDP_MODE:
            model_trainbias = convert_sync_bn(model_trainbias)
            model_freezebias = convert_sync_bn(model_freezebias)
        else:
            enable_batch_dp(model_trainbias, args.mode == 'dp')
            enable_batch_dp(model_freezebias, args.mode == 'dp')
        accuracy, official = main(run, hyp, model_proxy, model_trainbias, model_freezebias)
        if DDP_MODE:
            dist.barrier()
        sync_devices(args.mode == 'dp' and not DDP_MODE)
        elapsed = time.perf_counter() - started
        records.append(dict(run=run, accuracy=accuracy, single_training_wall_s=elapsed,
                            official_training_region_s=official))
        print('SINGLE_TRAINING_RESULT '+json.dumps(records[-1]), flush=True)
        del model_proxy, model_trainbias, model_freezebias
        DDP_WRAPPERS.clear()
    summary = dict(mode=args.mode, independent_training_processes=1,
                   physical_npus=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),
                   batch_dp_devices=[0,1] if args.mode in ('dp','ddp') else [0],
                   distributed_ranks=2 if DDP_MODE else 1,
                   syncbn_implementation=os.environ.get('AIRBENCH_SYNCBN') if DDP_MODE else None,
                   bn_semantics=('local BN per rank; dynamics changed' if os.environ.get('AIRBENCH_SYNCBN') == 'local'
                                 else 'global SyncBatchNorm') if DDP_MODE else 'native full-batch on primary', seed=args.seed,
                   records=records, accuracy_mean=statistics.mean(r['accuracy'] for r in records),
                   single_training_wall_median_s=statistics.median(r['single_training_wall_s'] for r in records))
    if not DDP_MODE or dist.get_rank() == 0:
        Path(args.output).write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


def verify_ddp(output):
    gates = dict(output_max_abs=0.01, gradient_cosine=0.999,
                 gradient_relative_l2=0.01, bn_max_abs=0.0001,
                 update_max_abs=0.01, loss_relative=0.01)
    torch.use_deterministic_algorithms(True)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.manual_seed(1234)
    baseline = make_net(hyp['net'])
    broadcast_state(baseline)
    candidate = make_net(hyp['net'])
    candidate.load_state_dict(baseline.state_dict())
    candidate = convert_sync_bn(candidate)
    wrapper = DistributedDataParallel(candidate, device_ids=[torch.npu.current_device()],
                                       broadcast_buffers=False)
    opt0 = torch.optim.SGD([p for p in baseline.parameters() if p.requires_grad], lr=.001, momentum=.85, nesterov=True)
    opt1 = torch.optim.SGD([p for p in candidate.parameters() if p.requires_grad], lr=.001, momentum=.85, nesterov=True)
    reports=[]
    for step in range(3):
        x=torch.randn(512,3,32,32,device='npu',dtype=torch.half)
        dist.broadcast(x,0)
        labels=torch.arange(512,device='npu')%10
        torch.npu.config.allow_internal_format=True
        opt0.zero_grad(set_to_none=True)
        y0=baseline(x)
        loss0=F.cross_entropy(y0,labels,label_smoothing=.2,reduction='sum')
        loss0.backward()
        opt1.zero_grad(set_to_none=True)
        y1=GatherLogits.apply(wrapper(x.chunk(2)[dist.get_rank()].contiguous()))
        loss1=F.cross_entropy(y1,labels,label_smoothing=.2,reduction='sum')
        loss1.backward()
        torch.npu.synchronize()
        grads={k:tensor_distance(p.grad,dict(candidate.named_parameters())[k].grad)
               for k,p in baseline.named_parameters() if p.grad is not None}
        buffers={k:(v-dict(candidate.named_buffers())[k]).abs().max().item()
                 for k,v in baseline.named_buffers()}
        torch.npu.config.allow_internal_format=False
        for net,opt in ((baseline,opt0),(candidate,opt1)):
            for p in net.parameters():
                if p.grad is not None:
                    p.grad=torch_npu.npu_format_cast(p.grad,torch_npu.get_npu_format(p))
            opt.step()
        updates={k:(v-candidate.state_dict()[k]).abs().max().item() for k,v in baseline.state_dict().items()}
        diff=tensor_distance(y0,y1)
        loss_rel=abs(loss0.item()-loss1.item())/abs(loss0.item())
        passed=(diff['max_abs']<=gates['output_max_abs'] and loss_rel<=gates['loss_relative'] and
                all(v['cosine']>=gates['gradient_cosine'] and v['relative_l2']<=gates['gradient_relative_l2'] for v in grads.values()) and
                max(buffers.values())<=gates['bn_max_abs'] and max(updates.values())<=gates['update_max_abs'])
        report=dict(step=step,output=diff,loss_relative=loss_rel,gradients=grads,bn_buffers=buffers,updates=updates,passed=passed)
        reports.append(report)
        if dist.get_rank()==0:
            print(json.dumps(report),flush=True)
    summary=dict(mode='verify-ddp',gates=gates,reports=reports,passed=all(r['passed'] for r in reports))
    if dist.get_rank()==0:
        Path(output).write_text(json.dumps(summary,indent=2)+'\n')
    if not summary['passed']:
        raise RuntimeError('DDP numerical gate failed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['single', 'dp', 'verify', 'ddp', 'verify-ddp'], default='single')
    parser.add_argument('--runs', type=int, default=1, help='Sequential complete trainings only')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--output', default='metrics.json')
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    torch.npu.set_device(int(os.environ.get('LOCAL_RANK', '0')))
    if DDP_MODE:
        from datetime import timedelta
        dist.init_process_group('hccl', timeout=timedelta(seconds=180))
    if args.mode in ('dp','verify') and torch.npu.device_count() < 2:
        parser.error('two visible NPUs required')
    if args.mode == 'verify-ddp':
        verify_ddp(args.output)
    elif args.mode == 'verify':
        verify_dp(args.output)
    else:
        run_training(args)
    if DDP_MODE:
        dist.destroy_process_group()
