import numpy as np
import os
import json 
from plate_optim.project_directories import main_dir, experiment_dir

os.chdir(main_dir)
from codeutils.logger import init_train_logger, print_log

from torchinfo import summary
import wandb, time, torch, argparse, ast
from torchvision import transforms
from plate_optim.pattern_generation import StandardPlateDataset
from plate_optim.flow_matching.train_plates import train
from plate_optim.flow_matching.model.unet import UNetModel


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


def get_args(string_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config.yaml", type=str, help='path to config file')
    parser.add_argument('--dir', default="debug", type=str, help='save directory')
    parser.add_argument('--device', default="cuda", type=str, help='choose cuda or cpu')
    parser.add_argument('--fp16', choices=[True, False], type=lambda x: x == 'True', default=True, help='use gradscaling (True/False)')
    parser.add_argument('--compile', choices=[True, False], type=lambda x: x == 'True', default=True, help='compile network (True/False)')
    parser.add_argument('--seed', default="0", type=int, help='seed')
    parser.add_argument('--lengthscale', default="0.025", type=float, help='dataset param')

    parser.add_argument('--batch_size', default="128", type=int, help='batch_size')
    parser.add_argument('--valset_size', default="128", type=int, help='batch_size')
    parser.add_argument('--dataset_size', default="100000", type=int, help='batch_size')
    parser.add_argument('--epochs', default="100", type=int, help='epochs')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--resolution', default="[96, 128]", type=ast.literal_eval, help='resolution')

    if string_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(string_args)

    args.config = os.path.join(main_dir, "configs", args.config)
    args.dir_name = args.dir
    args.original_dir = os.path.join(experiment_dir, args.dir)
    args.dir = os.path.join(experiment_dir, args.dir, time.strftime('%Y%m%d_%H%M%S', time.localtime()))
    os.makedirs(args.dir, exist_ok=True)
    return args

def norm(x):
    return x /0.02 * 2.0 - 1.0


def main():
    args = get_args()
    if args.debug is False:
        logger = init_train_logger(args)
        start_wandb(args)
    else:
        logger = None

    config = {
        "in_channels": 1,
        #"model_channels": 128,
        "model_channels": 64,
        "out_channels": 1,
        "num_res_blocks": 5,
        "attention_resolutions": [3, 4],
        "dropout": 0.1,
        #"channel_mult": [1, 1, 2],
        "channel_mult": [1, 1, 2, 2, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 4,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
    }

    json.dump(config, open(os.path.join(args.dir, "model_config.json"), "w"))
    model = UNetModel(**config).cuda()


    transform = transforms.Compose([
        transforms.Resize(args.resolution),
        norm
    ])
    dataset = StandardPlateDataset(transform=transform, size=args.dataset_size, min_length_scale=args.lengthscale)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    validation_dataset = StandardPlateDataset(transform=transform, size=args.valset_size, min_length_scale=args.lengthscale)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.valset_size)

    batch = next(iter(dataloader))
    with torch.no_grad():
        model(batch[0].cuda(), batch[1].cuda().squeeze(1), extra={})
        print_log(summary(model, input_data=[batch[0].cuda(), batch[1].cuda().squeeze(1), {}]), logger=logger)
    print_log(f' batch shape:{batch[0].shape}, conditioning shape {batch[1].shape}', logger=logger)
    print_log(f'Sample min_max {batch[0].min()}, {batch[0].max()}', logger=logger)

    if args.debug is True:
        return

    # prepare training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                total_iters=args.epochs,
                start_factor=1.0,
                end_factor=1e-8 / 1e-4,
            )

    train(args, model, dataloader, optimizer, lr_scheduler, logger, validation_dataloader)


def start_wandb(args):
    wandb.init(project='flow-matching', name=args.dir_name + "_" + str(time.time()), config=args)

if __name__ == '__main__':
    main()
