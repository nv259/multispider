import argparse
import functools
import collections
import datetime
import itertools
import json
import os
import traceback
from typing import Type, List

import _jsonnet
import torch
import numpy as np

# noinspection PyUnresolvedReferences
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

# noinspection PyUnresolvedReferences
from duorat import datasets

# noinspection PyUnresolvedReferences
from duorat import preproc

# noinspection PyUnresolvedReferences
from duorat import models

# noinspection PyUnresolvedReferences
from duorat.asdl.lang.spider.spider_transition_system import SpiderTransitionSystem
from duorat.types import RATPreprocItem

# noinspection PyUnresolvedReferences
from duorat.utils import optimizers
from duorat.utils import registry, parallelizer
from duorat.utils import random_state
from duorat.utils import saver as saver_mod
from duorat.utils.evaluation import evaluate_default, load_from_lines
from third_party.spider.evaluation import LEVELS
from preprocess import Preprocessor

from train import Logger, Trainer


class DEMATrainer(Trainer):    
    def load_train_config(self):
        # TODO: inspect config file
        
        if self.train_config.n_grad_accumulation_steps > 1: 
            self.logger.warn("Batch accumulation is used only at MAML-step level")
            raise NotImplementedError
   
    @staticmethod
    def get_kernel(params: torch.Tensor, num_particles):
        pass
    
    @staticmethod
    def get_kernel_wSGLD_B(params: torch.Tensor, num_particles):
        pass
    
    @staticmethod
    def get_pairwise_distance_matrix(x: torch.Tensor) -> torch.Tensor:
        pass
    
    @staticmethod
    def vector_to_list_params(vector, other_params):
        pass
    
    def ensemble_train(self, batch, prior_scale=1e-3):
        assert self.model.training
        
        inner_inter_params = []
        for i in range(self.num_particles):
            inner_inter_params.append(list(self.model.list_first_rat[i].parameters()))
            
        params_matrix = torch.stack(
            [torch.nn.utils.parameters_to_vector(params) for params in inner_inter_params],
            dim=0
        )
        
        pass
        
    def _update(self, train_data_loader, optimizer, lr_scheduler, scaler, saver, modeldir, last_step, prior_scale=1e-3):
        # Counter for grad aggregation
        grad_accumulation_counter = 0
        losses = []
        
        # 4. Start training loop
        with self.data_random:
            while True:
                # Quit if too long
                if last_step >= self.config["train"]["max_steps"]:
                    break
                
                # Compute and apply gradient
                with self.model_random:
                    # Assert all params have grad
                    for p in self.model.parameters():
                        p.grad = torch.zeros_like(p) if p.grad is None
                    
                    # Accumulate and update gradient 
                    for _ in range(self.config["train"]["n_grad_accumulation_steps"]):
                        batch = next(train_data_loader)
                        ensemble_train(batch, prior_scale)
        
        pass
     
    
def main(
        args=None, logdir_suffix: List[str] = None, trainer_class: Type[Trainer] = Trainer
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required='output/xlmr-multilingual')
    parser.add_argument("--config", default='configs/duorat/duorat-xlmr-multilingual.jsonnet')
    parser.add_argument("--preproc_data_path", default='dataset/pkl/xlmr-multilingual') 
    parser.add_argument("--load_path", default='')
    parser.add_argument("--step", default='')
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
    if args.load_path != '':
        trainer.train(modeldir=args.logdir, load_path=args.load_path, step=step)
    else:
        trainer.train(modeldir=args.logdir, step=step)

if __name__ == "__main__":
    main()
