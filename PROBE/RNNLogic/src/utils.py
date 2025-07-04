import sys
import os
import logging
import argparse
import random
import json
import yaml
import easydict
import numpy as np
import torch

def load_config(cfg_file):
    with open(cfg_file, "r") as fin:
        raw_text = fin.read()

    if "---" in raw_text:
        configs = []
        grid, template = raw_text.split("---")
        grid = yaml.safe_load(grid)
        template = jinja2.Template(template)
        for hyperparam in meshgrid(grid):
            config = easydict.EasyDict(yaml.safe_load(template.render(hyperparam)))
            configs.append(config)
    else:
        configs = [easydict.EasyDict(yaml.safe_load(raw_text))]

    return configs

def save_config(cfg, path):
    with open(os.path.join(path, 'config.yaml'), 'w') as fo:
        yaml.dump(dict(cfg), fo)

def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def save_model(model, optim, args):
    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    params = {
        'model': model.state_dict(),
        'optim': optim.state_dict()
    }

    torch.save(params, os.path.join(args.save_path, 'checkpoint'))

def load_model(model, optim, args):
    checkpoint = torch.load(args.load_path)
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])

def set_logger(args, save_path):
    log_file = os.path.join(save_path, f'run{args.seed}_{args.smoothing_p}(sm_p)_{args.smoothing_pp}(sm_pp).log')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)