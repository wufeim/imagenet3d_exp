import argparse
import json
import logging
import os

import numpy as np
from omegaconf import OmegaConf
import torch
import wandb

from imagenet3d.datasets import build_dataloader
from imagenet3d.utils import setup_logging, is_main_process, load_config, save_config, construct_class_by_name, AverageMeter
from imagenet3d.utils import bin_to_continuous, pose_error


def parse_args():
    parser = argparse.ArgumentParser(description='Pose estimation main script')
    parser.add_argument('--exp_name', type=str, default='pose_estimation_resnet50')
    parser.add_argument('--config', type=str, default='configs/pose_estimation_resnet50.yaml')
    parser.add_argument('--output_dir', type=str, default='exp')
    parser.add_argument('--known_classes', type=str, default='known_categories.txt')
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, scheduler, epo, cfg):
    model.train()

    loss_meter = AverageMeter('loss', 3)
    lr_meter = AverageMeter('lr', 5)

    for idx, sample in enumerate(dataloader):
        img = sample['img'].cuda(non_blocking=True)
        azimuth_idx = sample['azimuth_idx'].cuda(non_blocking=True)
        elevation_idx = sample['elevation_idx'].cuda(non_blocking=True)
        theta_idx = sample['theta_idx'].cuda(non_blocking=True)
        targets = torch.stack([azimuth_idx, elevation_idx, theta_idx], dim=1).view(-1)

        outputs = model(img).view(-1, cfg.multi_bin.num_bins)

        loss = construct_class_by_name(**cfg.training.loss)(
            outputs.view(-1, cfg.multi_bin.num_bins), targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item())
        lr_meter.update(optimizer.param_groups[0]['lr'])

        if is_main_process() and ((idx + 1) % cfg.training.log_freq == 0 or idx == len(dataloader) - 1):
            log_str = (f'[Epoch {epo+1:>{len(str(cfg.training.epochs))}}/{cfg.training.epochs} - '
                       f'{idx+1:>{len(str(len(dataloader)))}}/{len(dataloader)}] '
                       f'{str(loss_meter)} {str(lr_meter)} ')
            logging.info(log_str)
            if cfg.wandb.enabled:
                wandb.log({'loss': loss_meter.avg, 'lr': lr_meter.avg})

            loss_meter.reset()
            lr_meter.reset()

    scheduler.step()


@torch.no_grad()
def evaluate(model, dataloader, cfg, epo):
    model.eval()

    num_bins = cfg.multi_bin.num_bins

    pose_errors, categories, sample_names = [], [], []
    for i, sample in enumerate(dataloader):
        img = sample['img'].cuda(non_blocking=True)
        azimuth = sample['azimuth']
        elevation = sample['elevation']
        theta = sample['theta']

        categories += sample['cate_name']
        sample_names += sample['name']

        outputs = model(img)

        azimuth_pred = bin_to_continuous(np.argmax(outputs[:, 0:num_bins].detach().cpu().numpy(), axis=-1), **cfg.multi_bin)
        elevation_pred = bin_to_continuous(np.argmax(outputs[:, num_bins:2*num_bins].detach().cpu().numpy(), axis=-1), **cfg.multi_bin)
        theta_pred = bin_to_continuous(np.argmax(outputs[:, 2*num_bins:3*num_bins].detach().cpu().numpy(), axis=-1), **cfg.multi_bin)

        for j in range(img.shape[0]):
            pose_errors.append(pose_error(
                {'azimuth': azimuth[j], 'elevation': elevation[j], 'theta': theta[j]},
                {'azimuth': azimuth_pred[j], 'elevation': elevation_pred[j], 'theta': theta_pred[j]}).real)

    save_dict = {
        'sample_names': sample_names,
        'categories': categories,
        'pose_errors': pose_errors}
    with open(os.path.join(args.output_dir, f'evaluate_epo{epo+1}.json'), 'w') as fp:
        json.dump(save_dict, fp)

    pose_errors = np.array(pose_errors)
    pi_6_acc = np.mean(pose_errors < np.pi/6)
    pi_18_acc = np.mean(pose_errors < np.pi/18)
    med_err = np.median(pose_errors) / np.pi * 180.0
    mean_err = np.mean(pose_errors) / np.pi * 180.0

    if is_main_process():
        log_str = f'[Evaluate at epoch {epo+1}] pi_6_acc = {pi_6_acc*100:.2f}%, pi_18_acc = {pi_18_acc*100:.2f}%, med_err = {med_err:.2f}, mean_err = {mean_err:.2f}'
        logging.info(log_str)


def main(args):
    if is_main_process():
        setup_logging(args.output_dir, log_name='pose_estimation')

    with open(args.known_classes) as fp:
        known_classes = fp.read().strip().split('\n')

    cfg = load_config(args.config)
    cfg.exp_name = args.exp_name
    cfg.output_dir = args.output_dir
    cfg.data.train.categories = known_classes
    if is_main_process():
        logging.info(f'Configuration:\n{OmegaConf.to_yaml(cfg)}')
        save_config(cfg, os.path.join(args.output_dir, 'config.yaml'))

    if is_main_process() and cfg.wandb.enabled and not args.dry_run:
        run = wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.wandb.project,
            name=cfg.exp_name,
            reinit=True)

    logging.info('Building training and validation datasets...')
    train_dataloader, val_dataloader = build_dataloader(cfg, multi_bin=cfg.multi_bin)
    if is_main_process():
        logging.info('----------------')
        logging.info(f'Training dataset len: {len(train_dataloader.dataset)}')
        logging.info(f'Training dataloader len: {len(train_dataloader)}')
        logging.info(f'Train transform: {train_dataloader.dataset.transforms}')
        logging.info(f'Validation dataset len: {len(val_dataloader.dataset)}')
        logging.info(f'Validation dataloader len: {len(val_dataloader)}')
        logging.info(f'Validation transform: {val_dataloader.dataset.transforms}')
        logging.info('----------------')

    logging.info('Building pose estimation model')
    model = construct_class_by_name(**cfg.model)
    model = model.cuda()
    total_param, train_param = 0, 0
    for p in model.parameters():
        total_param += p.numel()
        if p.requires_grad:
            train_param += p.numel()
    logging.info(f'Total parameters = {total_param/1e6:.2f}M, trainable parameters = {train_param/1e6:.2f}M')

    logging.info('Building optimizer and scheduler')
    optimizer = construct_class_by_name(**cfg.training.optimizer, params=model.parameters())
    scheduler = construct_class_by_name(**cfg.training.scheduler, optimizer=optimizer)

    if args.dry_run:
        exit()

    logging.info('Start training...')
    for epo in range(cfg.training.epochs):
        train_one_epoch(model, train_dataloader, optimizer, scheduler, epo, cfg)

        if (epo + 1) % cfg.training.evaluate_freq == 0 or (epo == cfg.training.epochs - 1):
            evaluate(model, val_dataloader, cfg, epo)

        if (epo + 1) % cfg.training.checkpoint_freq == 0 or (epo == cfg.training.epochs - 1):
            os.makedirs(os.path.join(cfg.output_dir, 'ckpts'), exist_ok=True)
            torch.save(model.state_dict(), os.path.join(cfg.output_dir, 'ckpts', f'model_{epo+1}.pth'))


if __name__ == '__main__':
    args = parse_args()
    args.output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
