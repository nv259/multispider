import argparse
import collections
import datetime
import functools
import itertools
import json
import os
import traceback
from typing import List, Type

import _jsonnet
import numpy as np
import torch

# noinspection PyUnresolvedReferences
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
# noinspection PyUnresolvedReferences
from duorat import datasets, models, preproc

# noinspection PyUnresolvedReferences
from duorat.asdl.lang.spider.spider_transition_system import SpiderTransitionSystem
from duorat.types import RATPreprocItem

# noinspection PyUnresolvedReferences
from duorat.utils import optimizers, parallelizer, random_state, registry
from duorat.utils import saver as saver_mod
from duorat.utils.evaluation import evaluate_default, load_from_lines
from preprocess import Preprocessor
from third_party.spider.evaluation import LEVELS
from train import Logger, Trainer


class DEMATrainer(Trainer):
    def __init__(self, logger, config):
        super().__init__(logger, config)
        self.num_particles = self.config['model']['encoder']['num_particles']
         
    def get_kernel(self, params: torch.Tensor, num_particles):
        """
        Compute the RBF kernel for the input

        Args:
            params: a tensor of shape (N, M)

        Returns: kernel_matrix = tensor of shape (N, N)
        """
        pairwise_d_matrix = self.get_pairwise_distance_matrix(x=params)

        median_dist = torch.quantile(
            input=pairwise_d_matrix, q=0.5
        )  # tf.reduce_mean(euclidean_dists) ** 2
        h = median_dist / np.log(num_particles)

        kernel_matrix = torch.exp(-pairwise_d_matrix / h)
        kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
        grad_kernel = -torch.matmul(kernel_matrix, params)
        grad_kernel += params * kernel_sum
        grad_kernel /= h

        return kernel_matrix, grad_kernel, h

    def get_kernel_wSGLD_B(self, params: torch.Tensor, num_particles):
        """
        Compute the RBF kernel and repulsive term for wSGLD

        Args:
            params: a tensor of shape (N, M)

        Returns:
            - kernel_matrix = tensor of shape (N, N)
            - repulsive term
        """
        pairwise_d_matrix = DEMATrainer.get_pairwise_distance_matrix(x=params)
        median_dist = torch.quantile(input=pairwise_d_matrix, q=0.5)
        h = median_dist / np.log(num_particles)
        kernel_matrix = torch.exp(-pairwise_d_matrix / h)
        kernel_sum = torch.sum(input=kernel_matrix, dim=1, keepdim=True)
        # compute repulsive term of w_SGLD_B
        # invert of kernel_sum Nx1
        invert_kernel_sum = kernel_sum.pow_(-1)
        grad_kernel = params * (
            torch.matmul(kernel_matrix, invert_kernel_sum)
            + torch.sum(kernel_matrix * invert_kernel_sum, dim=1, keepdim=True)
        )
        grad_kernel += -(
            torch.matmul(
                kernel_matrix * torch.transpose(invert_kernel_sum, 0, 1), params
            )
            + torch.matmul(kernel_matrix, params) * invert_kernel_sum
        )
        grad_kernel /= h

        return kernel_matrix, grad_kernel, h

    def get_pairwise_distance_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the pairwise distance between each row of tensor x

        Args:
            x: input tensor

        Return: matrix of point-wise distances
        """
        n, m = x.shape

        # initialize matrix of pairwise distances as a N x N matrix
        pairwise_d_matrix = torch.zeros(size=(n, n), device=x.device)

        # num_particles = particle_tensor.shape[0]
        euclidean_dists = torch.nn.functional.pdist(input=x, p=2)  # shape of (N)

        # assign upper-triangle part
        triu_indices = torch.triu_indices(row=n, col=n, offset=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        # assign lower-triangle part
        pairwise_d_matrix = torch.transpose(pairwise_d_matrix, dim0=0, dim1=1)
        pairwise_d_matrix[triu_indices[0], triu_indices[1]] = euclidean_dists

        return pairwise_d_matrix

    def vector_to_list_params(self, vector, other_params):
        params = []

        # pointer for each layer params
        pointer = 0

        for param in other_params:
            # total number of params each layer
            num_params = int(np.prod(param.shape))

            params.append(vector[pointer : pointer + num_params].view(param.shape))

            pointer += num_params

        return params

    def ensemble_train(self, batch, prior_scale=1e-3):
        assert self.model.training

        inner_inter_params = []
        for i in range(self.num_particles):
            inner_inter_params.append(list(self.model.list_first_rats[i].parameters()))

        params_matrix = torch.stack(
            [
                torch.nn.utils.parameters_to_vector(params)
                for params in inner_inter_params
            ],
            dim=0,
        )

        particle_len = len(list(self.model.list_first_rats[0].parameters()))

        # For computing distance
        distance_nll = torch.empty(
            size=(self.num_particles, params_matrix.size(1)),
            device=next(self.model.parameters()).device,
        )

        final_losses = []

        for i in range(self.model.num_particles):
            loss = self.model.compute_branch_loss(batch, particle_idx=i)
            final_losses.append(loss.item())
            loss /= self.config["train"]["n_grad_accumulation_steps"]

            params_list = list(self.model.list_first_rats[i].parameters()) + list(
                param
                for name, param in self.model.named_parameters()
                if "list_first_rats" not in name
            )

            grads = torch.autograd.grad(loss, params_list, allow_unused=True)

            particle_grads = grads[:particle_len]
            model_wo_particle_grads = grads[particle_len:]

            distance_nll[i, :] = torch.nn.utils.parameters_to_vector(particle_grads)

        kernel_matrix, grad_kernel, _ = self.get_kernel(
            params=params_matrix,
            num_particles=self.config["train"]["n_grad_accumulation_steps"],
        )

        # Compute inner gradients with RPF kernel
        # SVGD + prior_scale * params_matrix
        # wSGLD_B
        # encoders_grads = distance_nll - grad_kernel
        encoders_grads = (1 / self.model.num_particles) * (
            torch.matmul(kernel_matrix, distance_nll) - grad_kernel
        )

        # Copy inner_grads to main network
        for i in range(self.model.num_particles):
            for p_tar, p_src in zip(
                self.model.list_first_rats[i].parameters(),
                self.vector_to_list_params(
                    encoders_grads[i], self.model.list_first_rats[i].parameters()
                ),
            ):
                p_tar.grad.data.add_(p_src)  # TODO: divided by #samples if inner is BA

        # NOTE: shorten SVGD step
        model_wo_particle_params = [
            param
            for name, param in self.model.named_parameters()
            if "list_first_rats" not in name
        ]
        for p_tar, p_src in zip(model_wo_particle_params, model_wo_particle_grads):
            p_tar.grad.data.add_(
                1 / self.model.num_particles * p_src
                if p_src is not None
                else torch.zeros_like(p_tar)
            )

        # TODO: Free GPU memory
        return sum(final_losses) / self.model.num_particles

    def _update(
        self,
        train_data_loader,
        train_eval_data_loader, 
        val_data_loader,
        optimizer,
        lr_scheduler,
        scaler,
        saver,
        modeldir,
        last_step,
        best_val_all_exact,
        prior_scale=1e-3,
    ):
        losses = []
        
        # 4. Start training loop
        with self.data_random:
            while True:
                # Quit if too long
                if last_step >= self.config["train"]["max_steps"]:
                    break
                
                oom = False

                try:
                    # Compute and apply gradient
                    with self.model_random:
                        # Assert all params have grad
                        for p in self.model.parameters():
                            p.grad = torch.zeros_like(p) if p.grad is None else p.grad

                        # Accumulate and update gradient
                        for _ in range(self.config["train"]["n_grad_accumulation_steps"]):
                            batch = next(train_data_loader)
                            losses.append(self.ensemble_train(batch, prior_scale))

                        # TODO: inspect AMP Scaler
                        for param_group in optimizer.param_groups:
                            torch.nn.utils.clip_grad_norm_(
                                param_group["params"], self.config["train"]["clip_grad"]
                            )

                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.update_lr(last_step)

                    last_step += 1

                    self._report(last_step=last_step, losses=losses, optimizer=optimizer)
                    best_val_all_exact = self._evaluate(train_eval_data_loader=train_eval_data_loader,
                                                        val_data_loader=val_data_loader,
                                                        last_step=last_step, 
                                                        saver=saver, 
                                                        modeldir=modeldir, 
                                                        best_val_all_exact=best_val_all_exact)

                    # Reset the list of losses
                    losses = []
                
                except RuntimeError as e:
                    err_msg = str(e)
                    self.logger.log(f"Forward Failed: {err_msg}")
                    oom = True

                # TODO: too many refcount to model, hence, gc.collect() cannot perform in the desired way
                # TODO: inspect refcount id(self.model))
                if oom: 
                    # Save the checkpoint and load to CPU
                    saver.save(
                        modeldir,
                        last_step,
                        is_best=False,
                        best_validation_metric=best_val_all_exact
                    )
                    self.model.to('cpu')
                    del self.model
                    _optimizer_to(optimizer, 'cpu')
                    del optimizer, lr_scheduler
                    torch.cuda.empty_cache()
                    import gc; gc.collect()
                    
                    # Load again
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    # self.__init__(self.logger, self.config)
                    self.model = registry.construct(
                        "model", self.config["model"], preproc=self.model_preproc,
                    )
                    self.model.to(device=device)
                    optimizer, lr_scheduler, _ = self._load_optimizer(self.config)

                    last_step, best_val_all_exact = saver.restore(modeldir, step=last_step, map_location=device)
                    self.logger.log(f"Model restored, the last step is {last_step}, best val_all_exact is {best_val_all_exact}")
                   
                    # Rename to last_step 
                    os.rename(
                        os.path.join(modeldir, f"model_step_{last_step}"),
                        os.path.join(modeldir, f"model_last_step")
                    )  
                    # Remove the tmp_checkpoint
                    # os.unlink(os.path.join(modeldir, f"model_step_{last_step}"))


def _optimizer_to(optimizer, device):
    "Move optimizer state to cpu"
    for param in optimizer.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def main(
    args=None,
    logdir_suffix: List[str] = None,
    trainer_class: Type[Trainer] = DEMATrainer,
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required="output/xlmr-multilingual")
    parser.add_argument(
        "--config", default="configs/duorat/dema-xlmr-multilingual.jsonnet"
    )
    parser.add_argument("--preproc_data_path", default="dataset/pkl/dema-multilingual")
    parser.add_argument("--load_path", default="")
    parser.add_argument("--step", default="")
    args = parser.parse_args(args)

    config = json.loads(_jsonnet.evaluate_file(args.config))

    if logdir_suffix:
        args.logdir = os.path.join(args.logdir, *logdir_suffix)

    if "model_name" in config:
        args.logdir = os.path.join(args.logdir, config["model_name"])

    # Initialize the logger
    reopen_to_flush = config.get("log", {}).get("reopen_to_flush")
    logger = Logger(os.path.join(args.logdir, "log.txt"), reopen_to_flush)
    logger.log("Logging to {}".format(args.logdir))
    logger.log(f"Overwriting preproc save_path with: {args.preproc_data_path}")

    if os.path.exists(args.preproc_data_path):
        logger.log("Skip preprocessing..")
    else:
        logger.log("Running preprocessing...")
        sections = config["data"].keys()
        keep_vocab = False
        preprocessor = Preprocessor(config)
        preprocessor.preprocess(sections, keep_vocab)

    # Construct trainer and do training
    step = None if not args.step else int(args.step)
    trainer = trainer_class(logger, config)
    if args.load_path != "":
        trainer.train(modeldir=args.logdir, load_path=args.load_path, step=step)
    else:
        trainer.train(modeldir=args.logdir, step=step)


if __name__ == "__main__":
    main()
