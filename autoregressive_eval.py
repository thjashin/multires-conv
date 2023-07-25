'''
Evaluate a multires conv autoregressive model from a checkpoint
'''
import os
import argparse
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
torch.backends.cuda.matmul.allow_tf32 = True
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

from autoregressive import MultiresAR
from tqdm.auto import tqdm
from utils import (
    ddp_setup, 
    count_parameters, 
    discretized_mix_logistic_loss, 
    sample_from_discretized_mix_logistic,
    DistributedSamplerNoDuplicate,
)


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
                    "Batch Idx: ({}/{}) | Loss: {:.4f}".format(batch_idx, len(dataloader), eval_bpd)
                )

    return eval_loss, total
    # return eval_loss / (total * np.prod(data_shape) * np.log(2.))


def sample(device, model, sample_size, data_shape):
    model.eval()
    with torch.no_grad():
        samples = torch.zeros(sample_size, data_shape[0], data_shape[1] * data_shape[2]).to(device)
        for i in range(data_shape[-2]):
            for j in tqdm(range(data_shape[-1])):
                pixel_loc = i * data_shape[-1] + j
                out = model(samples)
                out = sample_from_discretized_mix_logistic(out.reshape(*out.shape[:2], *data_shape[1:]), model.module.nr_logistic_mix)
                samples[:, :, pixel_loc] = out[:, :, i, j]
    return inv_rescaling(samples.reshape(-1, *data_shape))


def inv_rescaling(x):
    return 0.5 * x + 0.5

def rescaling(x):
    return (x - 0.5) * 2.

def image2seq(x):
    return x.view(3, 1024)


def main(rank, world_size, args):
    ddp_setup(rank, world_size, args.port)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'cifar10':
        transform = transforms.Compose([transforms.ToTensor(), rescaling, image2seq])

        testset = torchvision.datasets.CIFAR10(
            root='./data/cifar/', train=False, download=True, transform=transform)

        d_input = 3
        data_shape = (3, 32, 32)
        
        distributed_testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, 
            pin_memory=True, sampler=DistributedSamplerNoDuplicate(testset, shuffle=False))

        encoder = "linear"
        n_tokens = None
        max_length = 1024
    else: 
        raise NotImplementedError()

    torch.cuda.set_device(rank)
    torch.backends.cudnn.benchmark = False

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

    assert args.ckpt is not None
    log_dir = 'logs/ar/{}/{}'.format(args.dataset, args.ckpt)
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    # configure map_location properly
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    checkpoint = torch.load('{}/ckpt.pth'.format(log_dir), map_location=map_location)
    model.load_state_dict(checkpoint['model'])
    epoch = checkpoint['epoch']

    if rank == 0:
        logger = logging.getLogger(args.dataset)
        logger.setLevel(logging.INFO)
        log_path = os.path.join(log_dir, "eval.txt")
        info_file_handler = logging.FileHandler(log_path)
        console_handler = logging.StreamHandler()
        logger.addHandler(info_file_handler)
        logger.addHandler(console_handler)
        logger.info("Total number of parameters: {}".format(count_parameters(model)))

    test_loss_sum, total = eval(rank, distributed_testloader, model, criterion, data_shape)
    metrics = torch.tensor([test_loss_sum, total], dtype=torch.float64).to(rank)
    dist.all_reduce(metrics, dist.ReduceOp.SUM, async_op=False)
    if rank == 0:
        test_loss1 = metrics[0].item() / (metrics[1].item() * np.prod(data_shape) * np.log(2.))
        logger.info("{}, {}".format(metrics[0].item(), metrics[1].item()))
        logger.info("multi gpu result, epoch {}: test bits per dim={}".format(epoch, test_loss1))

        logger.info("Start saving samples...")
        samples = sample(rank, model, 64, data_shape)
        save_image(samples, '{}/images.png'.format(log_dir), nrow=8, padding=0)

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
    parser.add_argument('--ckpt', default=None, type=str, help='The log directory to resume from checkpoint')
    parser.add_argument('--port', default='12669', type=str, help='data parallel port')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')

    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size)
