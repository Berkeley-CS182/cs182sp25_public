import argparse
import numpy as np
import os
# import tensorflow as tf
import torch
import torchvision
import utils
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
from tqdm import tqdm



class Trainer:
    def __init__(self, model, optimizers, device="cuda",
        iter_max=10000, iter_save=1000, num_latents=100,
        modeltype="vae", writer=None, out_dir=""):
        self.model = model
        self.optimizers = optimizers
        self.device = device
        self.iter_save = iter_save
        self.iter_max = iter_max
        self.modeltype = modeltype
        self.writer = writer
        self.out_dir = out_dir

        self.step_fn = self._get_step_fn()

        # fix visualization latents
        self.z_test = torch.randn(100, num_latents).to(device)


    def build_input(self, x, y):
        if "vae" in self.model.name:
            x_real = torch.bernoulli(
                x.to(self.device).reshape(x.size(0), -1))
            y_real = y.new(np.eye(10)[y]).to(self.device).float()
        elif "gan" in self.model.name:
            x_real = x.to(self.device)
            y_real = y.to(self.device)
        return x_real, y_real

    def viz(self, global_step=1):
        with torch.no_grad():
            if "vae" in self.model.name:
                x_test = (self.model.sample_x(100) + 1) / 2.
                x_test = x_test.reshape(100, 1, 28, 28)
            elif "gan" in self.model.name:
                generator = self.model.g
                generator.eval()
                x_test = (generator(self.z_test) + 1) / 2.
                generator.train()
        torchvision.utils.save_image(
            x_test, '%s/generated_%s_%04d.png' % (self.out_dir, self.model.name.split('_')[0][-3:], global_step), nrow=10)


    def checkpoint_and_log(self, global_step, loss, summaries):
        if global_step % self.iter_save == 0:
            utils.log_summaries(self.writer, summaries, global_step)
            with torch.no_grad():
                self.viz(global_step)
                if "gan" in self.model.name:
                    torch.save((self.model.g, self.model.d), '%s/model_%04d.pt' % (self.out_dir, global_step))
            
    def gan_step(self, x_real, y_real):
        # breakpoint()
        assert len(self.optimizers) == 2

        g, d = self.model.g, self.model.d
        g_opt, d_opt = self.optimizers

        d_loss, g_loss = self.model.loss_nonsaturating(x_real, device=self.device)

        g_opt.zero_grad()
        g_loss.backward(retain_graph=True)
        g_opt.step()

        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        self.optimizers = [g_opt, d_opt]
        return {"d_loss": d_loss, "g_loss": g_loss}, None


    def vae_step(self, x_real, y_real):
        opt = self.optimizers[0]
        opt.zero_grad()
        loss, summaries = self.model.loss(x_real)
        loss.backward()
        
        opt.step()
        self.optimizers = [opt]
        return {"loss": loss}, summaries

    
    def _get_step_fn(self):
        # breakpoint()
        if "vae" in self.model.name:
            step_fn = self.vae_step
        elif "gan" in self.model.name:
            step_fn = self.gan_step
        return step_fn


    def evaluate(self, labeled_dataset):
        if "vae" in self.model.name:
            utils.evaluate_lower_bound(self.model, labeled_dataset)
        
        self.viz(global_step=10000)
        print("Saved visualization to %s/generated_%s_%s.png" % (self.out_dir, self.model.name.split('_')[0][-3:], self.iter_max))

    def train(self, train_loader, reinit=False):
        global_step = 0

        # train model from scratch
        if reinit:
            self.model.apply(utils.init_weights)
        
        # train models for multiple epochs
        with tqdm(total=self.iter_max) as pbar:
            while global_step < self.iter_max:
                for batch_idx, (x, y) in enumerate(train_loader):
                    x_real, y_real = self.build_input(x, y)
                    loss, summaries = self.step_fn(x_real, y_real)
                    global_step += 1    
                    pbar.update(1)
                    self.checkpoint_and_log(global_step, loss, summaries)
                    if global_step >= self.iter_max:
                        break

