import argparse
import numpy as np
import torch
import tqdm
import utils as ut
from models import vae
from models import gan
from train import Trainer
from pprint import pprint
from torchvision import datasets, transforms
import os




def build_config_from_args(is_jupyter=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=str, default='vae', choices=['vae', 'gan'])
    parser.add_argument('--num_latents', type=int, default=64, help="Number of latent dimensions")
    parser.add_argument('--iter_max', type=int, default=10000, help="Number of training iterations")
    parser.add_argument('--iter_save', type=int, default=500, help="Save model every n iterations")
    parser.add_argument('--runid', type=int, default=0, help="Run ID. In case you want to run replicates")
    parser.add_argument('--train', type=int, default=1, help="Flag for training")
    parser.add_argument('--out_dir', type=str, default="results", help="Flag for output logging")
    if is_jupyter:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()
    return args


def get_model_name(layout):
    model_name = '_'.join([t.format(v) for (t, v) in layout])
    print('Model name:', model_name)
    return model_name

def build_model(config, name=None, device='cpu'):
    if config.model == 'vae':
        model = vae.VAE(z_dim=config.num_latents, name=name)
    elif config.model == 'gan':
        model = gan.GAN(z_dim=config.num_latents, name=name)
    else:
        raise NotImplementedError
    return model.to(device)


def build_optimizers(model, config):
    if config.model == 'vae':
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizers = [opt]
    elif config.model == 'gan':
        g_opt = torch.optim.Adam(model.g.parameters(), lr=1e-3)
        d_opt = torch.optim.Adam(model.d.parameters(), lr=1e-3)
        optimizers = [g_opt, d_opt]
    return optimizers


def experiment(config=None):
    # breakpoint()
    layout = [
        ('model={:s}', config.model),
        ('z={:02d}',  config.num_latents),
        ('run={:04d}', config.runid)
    ]
    model_name = get_model_name(layout)
    pprint(vars(config))
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, labeled_subset, _ = ut.get_mnist_data(device, use_test_subset=True)
    model = build_model(config, model_name, device)
    optimizers = build_optimizers(model, config)

    try:
        os.mkdir(config.out_dir)
    except FileExistsError:
        pass


    writer = ut.prepare_writer(model_name, overwrite_existing=True)
    trainer = Trainer(model, optimizers,
                        writer=writer, device=device,
                        iter_max=config.iter_max,
                        iter_save=config.iter_save,
                        num_latents=config.num_latents,
                        out_dir=config.out_dir)

    trainer.train(train_loader)
    # breakpoint()

    trainer.evaluate(labeled_subset)


if __name__ == '__main__':
    config = build_config_from_args()
    experiment(config=config)