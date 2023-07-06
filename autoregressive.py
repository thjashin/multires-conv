"""Train a multires conv autoregressive model."""
import sys
import os
import argparse
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
torch.backends.cuda.matmul.allow_tf32 = True
import torchvision
import torchvision.transforms as transforms
import transformers
import numpy as np

from layers.multireslayer import MultiresLayer
from tqdm.auto import tqdm
from utils import (
    count_parameters, 
    discretized_mix_logistic_loss, 
    DistributedSamplerNoDuplicate, 
    apply_norm,
    ddp_setup,
    split_train_val,
)


class MultiresAR(nn.Module):

    def __init__(
        self,
        d_input,
        nr_logistic_mix,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        batchnorm=False,
        encoder="linear",
        n_tokens=None, 
        layer_type="multires",
        max_length=None,
        hinit=None,
        depth=None,
        tree_select="fading",
        d_mem=None,
        kernel_size=2,
        indep_res_init=False,
    ):
        super().__init__()

        self.batchnorm = batchnorm
        self.max_length = max_length
        self.depth = depth
        if encoder == "linear":
            self.encoder = nn.Conv1d(d_input, d_model, 1)
        elif encoder == "embedding":
            self.encoder = nn.Embedding(n_tokens, d_model)
        self.activation = nn.GELU()

        # Stack sequence modeling layers as residual blocks
        self.seq_layers = nn.ModuleList()
        self.mixing_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        if batchnorm:
            norm_func = nn.BatchNorm1d
        else:
            norm_func = nn.LayerNorm

        for _ in range(n_layers):
            if layer_type == "multires":
                layer = MultiresLayer(
                    d_model, 
                    kernel_size=kernel_size, 
                    depth=depth,
                    wavelet_init=hinit,
                    tree_select=tree_select,
                    seq_len=max_length,
                    dropout=dropout,
                    memory_size=d_mem,
                    indep_res_init=indep_res_init,
                )
            else:
                raise NotImplementedError()
            self.seq_layers.append(layer)

            activation_scaling = 2
            mixing_layer = nn.Sequential(
                nn.Conv1d(d_model, activation_scaling * d_model, 1),
                nn.GLU(dim=-2),
                nn.Dropout1d(dropout),
                nn.Conv1d(d_model, d_model, 1),
            )

            self.mixing_layers.append(mixing_layer)
            self.norms.append(norm_func(d_model))

        # Linear layer maps to mixiture of Logistics parameters
        num_mix = 3 if d_input == 1 else 10
        self.d_output = num_mix * nr_logistic_mix
        self.decoder = nn.Conv1d(d_model, self.d_output, 1)

    def forward(self, x, **kwargs):
        """Input shape: [bs, d_input, seq_len]. """
        # conv: [bs, d_input, seq_len] -> [bs, d_model, seq_len]
        # embedding: [bs, seq_len] -> [bs, seq_len, d_model]
        # Shift input by 1 pixel to perform autoregressive modeling
        x = torch.nn.functional.pad(x[..., :-1], (1, 0), "constant", 0)
        x = self.encoder(x)
        if isinstance(self.encoder, nn.Embedding):
            x = x.transpose(-1, -2)

        for layer, mixing_layer, norm in zip(
                self.seq_layers, self.mixing_layers, self.norms):
            x_orig = x
            x = layer(x)
            x = mixing_layer(x)
            x = x + x_orig

            x = apply_norm(x, norm, self.batchnorm)

        # out: [bs, d_output, seq_len]
        out = self.decoder(x)
        return out


def setup_optimizer(model, lr, epochs, iters_per_epoch, warmup):
    params = model.parameters()
    optimizer = optim.Adam(params, lr=lr)
    # We use the cosine annealing schedule sometimes with a linear warmup
    total_steps = epochs * iters_per_epoch
    if warmup > 0:
        warmup_steps = warmup * iters_per_epoch
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    return optimizer, scheduler


def train(device, epoch, trainloader, model, optimizer, scheduler, criterion, data_shape):
    data_shape = list(data_shape)
    model.train()
    train_loss = 0
    total = 0
    trainloader.sampler.set_epoch(epoch)
    pbar = enumerate(trainloader)
    if device == 0:
        pbar = tqdm(pbar)
    for batch_idx, batch in pbar:
        inputs, _ = batch
        inputs = inputs.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(inputs.reshape([-1] + data_shape), outputs.reshape([-1, outputs.shape[1]] + data_shape[-2:]))
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        total += inputs.shape[0]
        if device == 0:
            bpd_factor = total * np.prod(data_shape) * np.log(2.)
            train_bpd = train_loss / bpd_factor
            pbar.set_description(
                'Epoch {} | Batch Idx: ({}/{}) | Loss: {:.4f} | LR: {:.5f}'
                .format(epoch, batch_idx, len(trainloader), train_bpd, scheduler.get_lr()[0])
            )
    return train_loss / (total * np.prod(data_shape) * np.log(2.))


def eval(device, dataloader, model, criterion, data_shape):
    data_shape = list(data_shape)
    model.eval()
    eval_loss = 0
    total = 0
    with torch.no_grad():
        if device == 0:
            pbar = tqdm(enumerate(dataloader))
        else:
            pbar = enumerate(dataloader)
        for batch_idx, batch in pbar:
            inputs, _ = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(inputs.reshape([-1] + data_shape), outputs.reshape([-1, outputs.shape[1]] + data_shape[-2:]))

            eval_loss += loss.item()
            total += inputs.size(0)

            if device == 0:
                bpd_factor = total * np.prod(data_shape) * np.log(2.)
                eval_bpd = eval_loss / bpd_factor
                pbar.set_description(
                    'Batch Idx: ({}/{}) | Loss: {:.4f}'
                    .format(batch_idx, len(dataloader), eval_bpd)
                )

    # return eval_loss / (total * np.prod(data_shape) * np.log(2.))
    return eval_loss, total


def rescaling(x):
    return (x - 0.5) * 2.

def image2seq(x):
    return x.view(3, 1024)


def main(rank, world_size, args):
    ddp_setup(rank, world_size, args.port)
    assert args.batch_size % world_size == 0
    per_device_batch_size = args.batch_size // world_size
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), rescaling, image2seq])

        trainset = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=True, download=True, transform=transform)
        trainset, valset = split_train_val(trainset, val_split=0.04)

        testset = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=False, download=True, transform=transform)

        d_input = 3
        data_shape = (3, 32, 32)

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
            pin_memory=True, sampler=DistributedSamplerNoDuplicate(testset, shuffle=False))

        valloader = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
            pin_memory=True, sampler=DistributedSamplerNoDuplicate(valset, shuffle=False))

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=per_device_batch_size, num_workers=args.num_workers, shuffle=False, 
            pin_memory=True, drop_last=True, sampler=DistributedSampler(trainset))
        encoder = "linear"
        n_tokens = None
        max_length = 1024

    else: 
        raise NotImplementedError()

    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True

    # Model
    model = MultiresAR(
        d_input=d_input,
        nr_logistic_mix=10,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        batchnorm=args.batchnorm,
        encoder=encoder,
        n_tokens=n_tokens,
        layer_type=args.layer,
        max_length=max_length,
        hinit=args.hinit,
        depth=args.depth,
        tree_select=args.tree_select,
        d_mem=args.d_mem,
        kernel_size=args.kernel_size,
        indep_res_init=args.indep_res_init,
    ).to(rank)
    if args.batchnorm:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank])

    criterion = discretized_mix_logistic_loss
    optimizer, scheduler = setup_optimizer(
        model, lr=args.lr, epochs=args.epochs, iters_per_epoch=len(trainloader), warmup=args.warmup)

    if args.resume is not None:
        log_dir = 'logs/ar/{}/{}'.format(args.dataset, args.resume)
        print('==> Resuming from checkpoint..')
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load('{}/ckpt.pth'.format(log_dir), map_location=map_location)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['val_loss']
        final_test_loss = checkpoint['test_loss']
    else:
        log_dir = 'logs/ar/{}/{}'.format(args.dataset, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        start_epoch = 0
        best_val_loss = np.inf
        final_test_loss = 0

    if rank == 0:
        logger = logging.getLogger(args.dataset)
        logger.setLevel(logging.INFO)
        log_path = os.path.join(log_dir, "log.txt")
        info_file_handler = logging.FileHandler(log_path)
        console_handler = logging.StreamHandler()
        logger.addHandler(info_file_handler)
        logger.addHandler(console_handler)
        logger.info("Total number of parameters: {}".format(count_parameters(model)))

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(rank, epoch, trainloader, model, optimizer, scheduler, criterion, data_shape)
        if (epoch + 1) % args.test_every == 0:
            val_loss_sum, val_total = eval(rank, valloader, model, criterion, data_shape)
            val_metrics = torch.tensor([val_loss_sum, val_total], dtype=torch.float64).to(rank)
            dist.all_reduce(val_metrics, dist.ReduceOp.SUM, async_op=False)
            val_loss = val_metrics[0].item() / (val_metrics[1].item() * np.prod(data_shape) * np.log(2.))

            test_loss_sum, total = eval(rank, testloader, model, criterion, data_shape)
            metrics = torch.tensor([test_loss_sum, total], dtype=torch.float64).to(rank)
            dist.all_reduce(metrics, dist.ReduceOp.SUM, async_op=False)
            test_loss = metrics[0].item() / (metrics[1].item() * np.prod(data_shape) * np.log(2.))

            if val_loss <= best_val_loss:
                best_val_loss = val_loss
                final_test_loss = test_loss
                if rank == 0:
                    state = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'train_loss': train_loss,
                        'test_loss': test_loss,
                        'epoch': epoch,
                        'val_loss': val_loss,
                    }
                    torch.save(state, os.path.join(log_dir, "ckpt.pth"))

        if rank == 0:
            if (epoch + 1) % args.test_every == 0:
                logger.info("{}, {}, {}, {}".format(epoch, train_loss, val_loss, test_loss))
            else:
                logger.info("{}, {}, {}, {}".format(epoch, train_loss, -1, -1))
            with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
                f.write('\n'.join(sys.argv[1:]))
    if rank == 0:
        logger.info("FINAL: test bits per dim={}".format(final_test_loss))
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--lr', default=0.003, type=float, help='Learning rate')
    parser.add_argument('--epochs', default=250, type=int, help='Training epochs')
    parser.add_argument('--warmup', default=0, type=int, help='Number of warmup epochs')
    # Data
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10'], type=str)
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers for dataloader')
    parser.add_argument('--batch_size', default=64, type=int, help='Total batch size')
    # Model
    parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
    parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
    parser.add_argument('--layer', default='multires', choices=['multires'], type=str, 
                        help='Sequence modeling layer type')
    parser.add_argument('--d_mem', default=None, type=int, 
                        help='memory size, must be None for tree_select=fading')    
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout rate')
    parser.add_argument('--batchnorm', action='store_true', 
                        help='Replace layernorm with batchnorm')
    parser.add_argument('--hinit', default=None, type=str, help='Wavelet init')
    parser.add_argument('--depth', default=None, type=int, help='depth of each layer')
    parser.add_argument('--tree_select', default="fading", choices=["uniform", "fading"], 
                        help="Which part of the tree as memory")
    parser.add_argument('--kernel_size', default=2, type=int, 
                        help='Filter size, only used when hinit=None')
    parser.add_argument('--indep_res_init', action='store_true', 
                        help="Initialize w2 indepdent of z size in <w, [z, x]> = w1 z + w2 x")
    # Others
    parser.add_argument('--test_every', default=1, type=int, help='Every x epochs to eval the model')
    parser.add_argument('--resume', default=None, type=str, 
                        help='The log directory to resume from checkpoint')
    parser.add_argument('--port', default='12669', type=str, help='data parallel port')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
