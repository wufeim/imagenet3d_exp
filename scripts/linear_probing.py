import argparse
import json
import logging
import os

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import wandb

from imagenet3d.datasets import build_dataloader
from imagenet3d.utils import setup_logging, is_main_process, load_config, save_config, construct_class_by_name, AverageMeter
from imagenet3d.utils import bin_to_continuous, pose_error, batch_pose_error


def parse_args():
    parser = argparse.ArgumentParser(description='Linear probing of self-supervised models')
    parser.add_argument('--exp_name', type=str, default='linear_probing_dinov2_vits14')
    parser.add_argument('--config', type=str, default='configs/linear_probing_dinov2_vits14.yaml')
    parser.add_argument('--output_dir', type=str, default='exp')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()


def train_one_epoch(model, dataloader, optimizer, scheduler, epo, cfg):
    # In linear probing setting, backbone always in eval() mode.
    # model.train()

    loss_meter = AverageMeter('loss', 3)
    lr_meter = AverageMeter('lr', 5)

    loss_func = construct_class_by_name(**cfg.training.loss)

    for idx, sample in enumerate(dataloader):
        img = sample['img'].cuda(non_blocking=True)
        azimuth_idx = sample['azimuth_idx'].cuda(non_blocking=True)
        elevation_idx = sample['elevation_idx'].cuda(non_blocking=True)
        theta_idx = sample['theta_idx'].cuda(non_blocking=True)

        outputs = model(img)

        loss = 0.0
        for k in outputs:
            loss += loss_func(outputs[k][0], azimuth_idx)
            loss += loss_func(outputs[k][1], elevation_idx)
            loss += loss_func(outputs[k][2], theta_idx)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # FIXME: scheduler step position hard coded
        scheduler.step()

        loss_meter.update(loss.item())
        lr_meter.update(optimizer.param_groups[-1]['lr'])

        if is_main_process() and ((idx + 1) % cfg.training.log_freq == 0 or idx == len(dataloader) - 1):
            log_str = (f'[Epoch {epo+1:>{len(str(cfg.training.epochs))}}/{cfg.training.epochs} - '
                       f'{idx+1:>{len(str(len(dataloader)))}}/{len(dataloader)}] '
                       f'{str(loss_meter)} {str(lr_meter)} ')
            logging.info(log_str)
            if cfg.wandb.enabled:
                wandb.log({'loss': loss_meter.avg, 'lr': lr_meter.avg})

            loss_meter.reset()
            lr_meter.reset()


@torch.no_grad()
def evaluate(model, dataloader, cfg, epo):
    # In linear probing setting, backbone always in eval() mode.
    # model.eval()

    pose_errors, categories, sample_names = {}, [], []
    for i, sample in enumerate(dataloader):
        img = sample['img'].cuda(non_blocking=True)
        azimuth = sample['azimuth']
        elevation = sample['elevation']
        theta = sample['theta']
        cates = sample['cate_name']
        names = sample['name']
        categories += cates
        sample_names += names

        outputs = model(img)

        for k in outputs:
            azimuth_pred = bin_to_continuous(np.argmax(outputs[k][0].detach().cpu().numpy(), axis=-1), **cfg.multi_bin)
            elevation_pred = bin_to_continuous(np.argmax(outputs[k][1].detach().cpu().numpy(), axis=-1), **cfg.multi_bin)
            theta_pred = bin_to_continuous(np.argmax(outputs[k][2].detach().cpu().numpy(), axis=-1), **cfg.multi_bin)
            if k not in pose_errors:
                pose_errors[k] = []
            pose_errors[k] += batch_pose_error(
                (azimuth.detach().cpu().numpy(), elevation.detach().cpu().numpy(), theta.detach().cpu().numpy()),
                (azimuth_pred, elevation_pred, theta_pred)).real.tolist()
            # for j in range(img.shape[0]):
            #     pose_errors[k].append(pose_error(
            #         {'azimuth': azimuth[j], 'elevation': elevation[j], 'theta': theta[j]},
            #         {'azimuth': azimuth_pred[j], 'elevation': elevation_pred[j], 'theta': theta_pred[j]}))

    save_dict = {
        'sample_names': sample_names,
        'categories': categories,
        'pose_errors': pose_errors}
    with open(os.path.join(args.output_dir, f'evaluate_epo{epo+1}.json'), 'w') as fp:
        json.dump(save_dict, fp)

    best_pi_6_acc, best_model_name = 0.0, None
    results = {}
    for k in pose_errors:
        _pose_errors = np.array(pose_errors[k])
        results[k] = dict(
            pi_6_acc=np.mean(_pose_errors < np.pi/6),
            pi_18_acc=np.mean(_pose_errors < np.pi/18),
            med_err=np.median(_pose_errors) / np.pi * 180.0,
            mean_err=np.mean(_pose_errors) / np.pi * 180.0)
        if results[k]['pi_6_acc'] > best_pi_6_acc:
            best_pi_6_acc = results[k]['pi_6_acc']
            best_model_name = k
    best_model_result = results[best_model_name]

    if is_main_process():
        log_str = (
            f'[Evaluate at epoch {epo+1}] best model {k}: '
            f'pi_6_acc = {best_model_result["pi_6_acc"]*100:.2f}%, '
            f'pi_18_acc = {best_model_result["pi_18_acc"]*100:.2f}%, '
            f'med_err = {best_model_result["med_err"]:.2f}, '
            f'mean_err = {best_model_result["mean_err"]:.2f}')
        logging.info(log_str)


def main(args):
    if is_main_process():
        setup_logging(args.output_dir, log_name='linear_probing')

    cfg = load_config(args.config)
    cfg.exp_name = args.exp_name
    cfg.output_dir = args.output_dir
    cfg.training.epochs = args.epochs
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
        logging.info(f'Validation transform: {train_dataloader.dataset.transforms}')
        logging.info('----------------')

    logging.info('Building linear probing model')
    model = construct_class_by_name(**cfg.model)
    model = model.cuda()
    total_param, train_param = 0, 0
    for p in model.parameters():
        total_param += p.numel()
        if p.requires_grad:
            train_param += p.numel()
    logging.info(f'Total parameters = {total_param/1e6:.2f}M, trainable parameters = {train_param/1e6:.2f}M')

    # FIXME: T_max is hard-coded here
    if (
        cfg.training.scheduler.class_name == 'torch.optim.lr_scheduler.CosineAnnealingLR'
        and cfg.training.scheduler.T_max == -1
    ):
        cfg.training.scheduler.T_max = cfg.training.epochs * len(train_dataloader)

    logging.info('Building optimizer and scheduler')
    optimizer = construct_class_by_name(**cfg.training.optimizer, params=model.optim_param_groups)
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
