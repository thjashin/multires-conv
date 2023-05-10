"""Classification code (multi-gpu)."""
import sys
import os
import argparse
from datetime import datetime
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
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
    apply_norm,
    ddp_setup,
    split_train_val,
)

from dataloaders.lra import ListOps


def masked_meanpool(x, lengths):
    # x: [bs, H, L]
    # lengths: [bs]
    L = x.shape[-1]
    # mask: [bs, L]
    mask = torch.arange(L, device=x.device) < lengths[:, None]
    # ret: [bs, H]
    return torch.sum(mask[:, None, :] * x, -1) / lengths[:, None]


class MultiresNet(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
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
        self.dropouts = nn.ModuleList()

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
            )

            self.mixing_layers.append(mixing_layer)
            self.norms.append(norm_func(d_model))

        # Linear layer maps to logits
        self.output_mapping = nn.Linear(d_model, d_output)

    def forward(self, x, **kwargs):
        """Input shape: [bs, d_input, seq_len]. """
        # conv: [bs, d_input, seq_len] -> [bs, d_model, seq_len]
        # embedding: [bs, seq_len] -> [bs, seq_len, d_model]
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

        # mean_pooling: [bs, d_model, seq_len] -> [bs, d_model]
        lengths = kwargs.get("lengths", None)
        if lengths is not None:
            lengths = lengths.to(x.device)
            # only pooling over the steps corresponding to actual inputs
            x = masked_meanpool(x, lengths)
        else:
            x = x.mean(dim=-1)

        # out: [bs, d_output]
        out = self.output_mapping(x)
        return out


def setup_optimizer(model, lr, weight_decay, epochs, iters_per_epoch, warmup):
    params = model.parameters()
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    # We use the cosine annealing schedule sometimes with a linear warmup
    total_steps = epochs * iters_per_epoch
    if warmup > 0:
        warmup_steps = warmup * iters_per_epoch
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    return optimizer, scheduler


def train(device, epoch, trainloader, model, optimizer, scheduler, criterion):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    trainloader.sampler.set_epoch(epoch)
    n_iters = len(trainloader)
    pbar = enumerate(trainloader)
    if device == 0:
        pbar = tqdm(pbar)
    for batch_idx, batch in pbar:
        inputs, targets, *z = batch
        if len(z) == 0:
            z = {}
        else:
            z = z[0]
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        num_pad = model.module.max_length - inputs.shape[-1]
        if num_pad > 0:
            # vocab padding value is zero
            inputs = nn.functional.pad(inputs, (0, num_pad), "constant", 0)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs, **z)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100 * correct / total
        if device == 0:
            pbar.set_description(
                'Epoch {} | Batch Idx: ({}/{}) | Loss: {:.3f} | Acc: {:.3f}%% | LR: {:.5f}'
                .format(epoch, batch_idx, len(trainloader), train_loss / (batch_idx + 1), acc, scheduler.get_lr()[0])
            )
    return train_loss / n_iters, 100. * correct / total


def eval(device, epoch, dataloader, model, criterion):
    model.eval()
    eval_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader))
        for batch_idx, batch in pbar:
            inputs, targets, *z = batch
            if len(z) == 0:
                z = {}
            else:
                z = z[0]
            inputs, targets = inputs.to(device), targets.to(device)
            num_pad = model.module.max_length - inputs.shape[-1]
            if num_pad > 0:
                # Assuming vocab padding value is zero
                inputs = nn.functional.pad(inputs, (0, num_pad), "constant", 0)
            outputs = model(inputs, **z)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_description(
                'Batch Idx: ({}/{}) | Loss: {:.3f} | Acc: {:.3f}%%'
                .format(batch_idx, len(dataloader), eval_loss / (batch_idx + 1), 100. * correct / total)
            )

    return 100. * correct / total


def image2seq(x):
    return x.view(3, 1024)

def image2seq_grayscale(x):
    return x.view(1, 1024)


def main(rank, world_size, args):
    ddp_setup(rank, world_size, args.port)
    assert args.batch_size % world_size == 0
    per_device_batch_size = args.batch_size // world_size
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Lambda(image2seq)
        ])

        trainset = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=True, download=True, transform=transform)
        trainset, valset = split_train_val(trainset, val_split=0.1)

        testset = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=False, download=True, transform=transform)

        d_input = 3
        d_output = 10

        valloader = torch.utils.data.DataLoader(
            valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=per_device_batch_size, num_workers=args.num_workers, shuffle=False, 
            pin_memory=True, sampler=DistributedSampler(trainset))
        encoder = "linear"
        n_tokens = None
        max_length = 1024

    elif args.dataset == "listops":
        listops = ListOps(args.dataset, data_dir='./data')
        listops.setup()
        valloader = listops.val_dataloader(
            batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=args.num_workers)
        testloader = listops.test_dataloader(
            batch_size=args.batch_size, drop_last=False, shuffle=False, num_workers=args.num_workers)
        trainloader = listops.train_dataloader(
            batch_size=per_device_batch_size, num_workers=args.num_workers, shuffle=False, 
            pin_memory=True, sampler=DistributedSampler(listops.dataset_train))
        d_input = 1
        d_output = listops.d_output
        encoder = "embedding"
        n_tokens = listops.n_tokens
        max_length = listops.l_max

    else: 
        raise NotImplementedError()

    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = True

    # Model
    model = MultiresNet(
        d_input=d_input,
        d_output=d_output,
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

    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = setup_optimizer(
        model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs, 
        iters_per_epoch=len(trainloader), warmup=args.warmup,
    )

    if args.resume is not None:
        log_dir = 'logs/{}/{}'.format(args.dataset, args.resume)
        print('==> Resuming from checkpoint..')
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
        checkpoint = torch.load('{}/ckpt.pth'.format(log_dir), map_location=map_location)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        best_val_acc = checkpoint['val_acc']
        final_test_acc = checkpoint['test_acc']
        start_epoch = checkpoint['epoch']
    else:
        log_dir = 'logs/{}/{}'.format(args.dataset, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        start_epoch = 0
        best_val_acc = 0
        final_test_acc = 0

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
        train_loss, train_acc = train(rank, epoch, trainloader, model, optimizer, scheduler, criterion)
        if rank == 0:
            val_acc = eval(rank, epoch, valloader, model, criterion)
            test_acc = eval(rank, epoch, testloader, model, criterion)
            logger.info("{}, {}, {}, {}, {}".format(epoch, train_loss, train_acc, val_acc, test_acc))

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc
                state = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'val_acc': val_acc,
                    'test_acc': test_acc,
                    'epoch': epoch,
                }

                torch.save(state, os.path.join(log_dir, "ckpt.pth"))
                with open(os.path.join(log_dir, 'args.txt'), 'w') as f:
                    f.write('\n'.join(sys.argv[1:]))
    if rank == 0:
        logger.info("FINAL: valid acc={}, test acc={}".format(best_val_acc, final_test_acc))
    destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float)
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    parser.add_argument('--warmup', default=0, type=int, help='Number of warmup epochs')
    # Data
    parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'listops'], type=str)
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers for dataloader')
    parser.add_argument('--batch_size', default=64, type=int, help='Total batch size')
    # Model
    parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
    parser.add_argument('--d_model', default=128, type=int, help='Number of channels')
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
    parser.add_argument('--resume', default=None, type=str, 
                        help='The log directory to resume from checkpoint')
    parser.add_argument('--port', default='12669', type=str, help='data parallel port')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
