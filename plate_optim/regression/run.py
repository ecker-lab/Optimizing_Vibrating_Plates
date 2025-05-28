import numpy as np
import os
from plate_optim.regression.regression_model import get_net
from plate_optim.regression.data import get_dataloader
from plate_optim.regression.train import train, evaluate
from plate_optim.project_directories import wandb_project, main_dir, experiment_dir

os.chdir(main_dir)
from codeutils.builder import build_opti_sche
from codeutils.logger import init_train_logger, print_log
from codeutils.config import get_config

from torchinfo import summary
import wandb, time, torch, argparse

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def get_args(string_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yaml", type=str, help='path to config file')
    parser.add_argument('--dir', default="debug", type=str, help='save directory')
    parser.add_argument('--device', default="cuda", type=str, help='choose cuda or cpu')
    parser.add_argument('--fp16', choices=[True, False], type=lambda x: x == 'True', default=True, help='use gradscaling (True/False)')
    parser.add_argument('--compile', choices=[True, False], type=lambda x: x == 'True', default=True, help='compile network (True/False)')
    parser.add_argument('--seed', default="0", type=int, help='seed')
    parser.add_argument('--add_noise', type=float, default=0, help='add noise to beading pattern images during training')

    parser.add_argument('--batch_size', default="64", type=int, help='batch_size')
    parser.add_argument('--scaling_factor', default="32", type=int, help='network size scaling')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--load_idx', action='store_true', help='debug mode')
    parser.add_argument('--continue_training', action='store_true', help='continue training from checkpoint')

    if string_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(string_args)

    args.config = os.path.join(main_dir, "configs", args.config)
    args.dir_name = args.dir
    args.original_dir = os.path.join(experiment_dir, args.dir)
    args.dir = os.path.join(experiment_dir, args.dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
    return args


def main():
    args = get_args()
    config = get_config(args.config)

    if args.debug is False:
        logger = init_train_logger(args, config)
        start_wandb(args, config)
    else:
        config.dataset_keys = ["bead_patterns", "z_vel_mean_sq", "phy_para", "frequencies"]
        logger = None

    net = get_net(conditional=True, len_conditional=3, scaling_factor=args.scaling_factor).to(args.device)
    optimizer, scheduler = build_opti_sche(net, config)
    trainloader, valloader, testloader, _, _, _ = get_dataloader(args, config, logger)
    batch = next(iter(trainloader))
    batch = {k: v.to(args.device) for k, v in batch.items()}
    print_log(summary(net, input_data=(batch["bead_patterns"], batch["phy_para"], batch["frequencies"])), logger=logger)

    if args.debug is True:
        return

    if hasattr(config, "initial_checkpoint"):
        print_log(f"loading checkpoint from {config.initial_checkpoint}")
        data = torch.load(config.initial_checkpoint)
        missing_keys, unexpected_keys = net.load_state_dict(data["model_state_dict"], strict=False)
        print_log(f"missing keys: {missing_keys}, unexpected keys: {unexpected_keys}", logger=logger)

    if args.continue_training:
        net.load_state_dict(data["model_state_dict"])
        start_epoch = data["epoch"]
        optimizer.load_state_dict(data["optimizer_state_dict"])
        lowest_loss = 100
        scheduler.step(start_epoch)
        print_log(f"Continue training from epoch {start_epoch}, with loss {lowest_loss}", logger=logger)
        net = train(args,
                    config,
                    net,
                    trainloader,
                    optimizer,
                    valloader,
                    scheduler=scheduler,
                    logger=logger,
                    start_epoch=start_epoch,
                    lowest_loss=lowest_loss)
    else:
        net = train(args, config, net, trainloader, optimizer, valloader, scheduler=scheduler, logger=logger)

    print_log(f"Evaluate on test set", logger=logger)
    results = evaluate(args, config, net, testloader, logger=logger, epoch=None)
    # a, b, c, save_rmean = results["loss (test/val)"], results["frequency_distance"], results["save_rmean"]
    # print_log(f"{a:4.2f} & {b:4.2f} & {save_rmean:3.2f} & {c:3.1f}", logger=logger)


def start_wandb(args, config):
    wandb.init(project=wandb_project)
    wandb.config.update(config)
    wandb.run.name = args.dir_name + "_" + str(time.time())


if __name__ == '__main__':
    main()
