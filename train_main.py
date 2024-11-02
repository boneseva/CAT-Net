#!/usr/bin/env python
"""
For Evaluation
Extended from ADNet code by Hansen et al.
"""
import os
import random
import logging
import shutil

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from models.fewshot import FewShotSeg
from dataloaders.datasets import TrainDataset as TrainDataset
from utils import *
from config import ex

import wandb  # Add wandb to your imports
from datetime import datetime

@ex.automain
def main(_run, _config, _log):
    if 'pretrain_path' not in _config:
        _config['pretrain_path'] = None

    # Initialize wandb
    wandb.init(
        project="LRDL-CATNet",
        name=f"Run_{_config['exp_str']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  # Example with a timestamp
        config=_config,
        entity="boneseva",
    )
    wandb.run.name = f"Run_{_config['exp_str']}"

    if _run.observers:
        # Set up source folder
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

        # Set up logger -> log to .txt
        file_handler = logging.FileHandler(os.path.join(f'{_run.observers[0].dir}', f'logger.log'))
        file_handler.setLevel('INFO')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(formatter)
        _log.handlers.append(file_handler)
        _log.info(f'Run "{_config["exp_str"]}" with ID "{_run.observers[0].dir[-1]}"')

    # Deterministic setting for reproducibility.
    if _config['seed'] is not None:
        random.seed(_config['seed'])
        torch.manual_seed(_config['seed'])
        torch.cuda.manual_seed_all(_config['seed'])
        cudnn.deterministic = True

    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)

    _log.info(f'Create model...')
    model = FewShotSeg().cuda()
    model.train()

    # load the pretrained model
    if _config['pretrain_path'] is not None:
        state_dict = torch.load(_config['pretrain_path'])
        model.load_state_dict(state_dict, strict=False)  # strict=False allows partial loading
        _log.info(f'Loaded pretrained model from: {_config["pretrain_path"]}')

    _log.info(f'Set optimizer...')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    lr_milestones = [(ii + 1) * _config['max_iters_per_load'] for ii in
                     range(_config['n_steps'] // _config['max_iters_per_load'] - 1)]
    scheduler = MultiStepLR(optimizer, milestones=lr_milestones, gamma=_config['lr_step_gamma'])

    my_weight = torch.FloatTensor([0.1, 1.0]).cuda()
    criterion = nn.NLLLoss(ignore_index=255, weight=my_weight)

    _log.info(f'Load data...')
    data_config = {
        'data_dir': _config['path'][_config['dataset']]['data_dir'],
        'dataset': _config['dataset'],
        'n_shot': _config['n_shot'],
        'n_way': _config['n_way'],
        'n_query': _config['n_query'],
        'n_sv': _config['n_sv'],
        'max_iter': _config['max_iters_per_load'],
        'eval_fold': _config['eval_fold'],
        'min_size': _config['min_size'],
        'max_slices': _config['max_slices'],
        'test_label': _config['test_label'],
        'exclude_label': _config['exclude_label'],
        'use_gt': _config['use_gt'],
        'pretrain_path': _config['pretrain_path'],
    }
    train_dataset = TrainDataset(data_config)
    train_loader = DataLoader(train_dataset,
                              batch_size=_config['batch_size'],
                              shuffle=True,
                              num_workers=_config['num_workers'],
                              pin_memory=True,
                              drop_last=True)

    n_sub_epochs = _config['n_steps'] // _config['max_iters_per_load']
    log_loss = {'total_loss': 0, 'query_loss': 0, 'align_loss': 0}

    i_iter = 0
    _log.info(f'Start training...')
    for sub_epoch in range(n_sub_epochs):
        _log.info(f'This is epoch "{sub_epoch}" of "{n_sub_epochs}" epochs.')
        for _, sample in enumerate(train_loader):
            support_images = [[shot.float().cuda() for shot in way] for way in sample['support_images']]
            support_fg_mask = [[shot.float().cuda() for shot in way] for way in sample['support_fg_labels']]
            query_images = [query_image.float().cuda() for query_image in sample['query_images']]
            query_labels = torch.cat([query_label.long().cuda() for query_label in sample['query_labels']], dim=0)

            query_pred, align_loss = model(support_images, support_fg_mask, query_images, train=True)
            query_loss = criterion(torch.log(torch.clamp(query_pred, torch.finfo(torch.float32).eps,
                                                         1 - torch.finfo(torch.float32).eps)), query_labels)
            loss = query_loss + align_loss

            for param in model.parameters():
                param.grad = None

            loss.backward()
            optimizer.step()
            scheduler.step()

            query_loss_val = query_loss.detach().data.cpu().numpy()
            align_loss_val = align_loss.detach().data.cpu().numpy()

            _run.log_scalar('total_loss', loss.item())
            _run.log_scalar('query_loss', query_loss_val)
            _run.log_scalar('align_loss', align_loss_val)

            # Prepare images for wandb logging
            ground_truth_img = sample['query_labels'][0][0].cpu().numpy()
            prediction_img = query_pred[0][0].cpu().detach().numpy()
            support_img = support_images[0][0][0].cpu().numpy()
            query_img = query_images[0][0].cpu().numpy()

            # Convert images to HWC format for wandb logging
            if ground_truth_img.ndim == 3:  # Multi-channel image
                ground_truth_img = ground_truth_img.transpose(1, 2, 0)  # CHW to HWC
            else:  # Single-channel image (H, W)
                ground_truth_img = ground_truth_img[:, :, None]  # Add a channel dimension to make it HWC

            if prediction_img.ndim == 3:  # Multi-channel image
                prediction_img = prediction_img.transpose(1, 2, 0)  # CHW to HWC
            else:  # Single-channel image
                prediction_img = prediction_img[:, :, None]

            if support_img.ndim == 3:  # Multi-channel image
                support_img = support_img.transpose(1, 2, 0)  # CHW to HWC
            else:  # Single-channel image
                support_img = support_img[:, :, None]

            if query_img.ndim == 3:  # Multi-channel image
                query_img = query_img.transpose(1, 2, 0)  # CHW to HWC
            else:  # Single-channel image
                query_img = query_img[:, :, None]

            # Log metrics and images to wandb
            wandb.log({
                "total_loss": loss.item(),
                "query_loss": query_loss_val,
                "align_loss": align_loss_val,
                "iteration": i_iter + 1,
                "ground_truth": wandb.Image(ground_truth_img, caption="Ground Truth"),
                "prediction": wandb.Image(prediction_img, caption="Prediction"),
                "support_image": wandb.Image(support_img, caption="Support Image"),
                "query_image": wandb.Image(query_img, caption="Query Image")
            })

            log_loss['total_loss'] += loss.item()
            log_loss['query_loss'] += query_loss_val
            log_loss['align_loss'] += align_loss_val

            if (i_iter + 1) % _config['print_interval'] == 0:
                total_loss = log_loss['total_loss'] / _config['print_interval']
                query_loss = log_loss['query_loss'] / _config['print_interval']
                align_loss = log_loss['align_loss'] / _config['print_interval']

                log_loss['total_loss'] = 0
                log_loss['query_loss'] = 0
                log_loss['align_loss'] = 0

                _log.info(f'step {i_iter + 1}: total_loss: {total_loss}, query_loss: {query_loss},'
                          f' align_loss: {align_loss}')

            if (i_iter + 1) % _config['save_snapshot_every'] == 0:
                _log.info('###### Taking snapshot ######')
                model_path = os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth')
                torch.save(model.state_dict(), model_path)
                wandb.save(model_path)  # Save model to wandb as an artifact

            i_iter += 1

    _log.info('End of training.')
    wandb.finish()  # End the wandb run
    return 1
