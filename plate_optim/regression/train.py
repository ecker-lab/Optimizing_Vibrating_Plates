import os
import numpy as np
import torch, wandb
import torch.nn.functional as F
from codeutils.logger import print_log
from plate_optim.regression.regression_model import get_mean_from_velocity_field
from plate_optim.utils.training import save_model
from plate_optim.regression.data import extract_mean_std

fields = ["bead_patterns", "z_vel_abs", "z_vel_mean_sq", "phy_para", "frequencies"]


def add_noise(image, add_noise=0):
    if add_noise != 0:
        noise_levels = (torch.rand(image.size(0), 1, 1, 1)).to(image.device) * add_noise
        noisy_image = (1 - noise_levels) * image + torch.randn_like(image) * noise_levels # this matches the OT noise formulation
    else:
        noisy_image = image
        noise_levels = torch.zeros(image.size(0), 1, 1, 1).to(image.device)
    return noisy_image, noise_levels.squeeze(1, 2)


def train(args, config, net, dataloader, optimizer, valloader, scheduler, logger=None, start_epoch=0, lowest_loss=np.inf):
    net.train()
    torch.set_float32_matmul_precision('high')
    if args.compile:
        net_ = torch.compile(net, dynamic=True)
    else:
        net_ = net

    scaler = torch.amp.GradScaler(device='cuda', enabled=args.fp16)
    out_mean, out_std, field_mean, field_std = extract_mean_std(dataloader.dataset)
    out_mean, out_std = out_mean.to(args.device), out_std.to(args.device)
    field_mean, field_std = field_mean.to(args.device), field_std.to(args.device)

    for epoch in range(start_epoch + 1, config.epochs):
        losses_freq, losses_field = [], []
        for batch in dataloader:
            optimizer.zero_grad()
            image, velocity_field, vel_mean_sq, condition, frequencies = (batch[field].to(args.device) for field in fields)
            image, noise_levels = add_noise(image, args.add_noise)
            with torch.amp.autocast(device_type='cuda', enabled=args.fp16):
                prediction = net_(image, condition, frequencies)
                loss = F.mse_loss(prediction, velocity_field)
            losses_field.append(loss.detach().cpu().item())
            scaler.scale(loss.mean()).backward()

            if config.optimizer.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), config.optimizer.gradient_clip)

            scaler.step(optimizer), scaler.update()

            with torch.no_grad():
                prediction = get_mean_from_velocity_field(prediction, field_mean, field_std, frequencies)
                prediction.sub_(out_mean[frequencies]).div_(out_std)
                loss_freq = F.mse_loss(prediction, vel_mean_sq, reduction='none')
                losses_freq.append(loss_freq.detach().cpu())
        if scheduler is not None:
            scheduler.step(epoch)

        print_log(f"Epoch {epoch} training loss = {(np.mean(losses_field)):4.4}, {(np.mean(losses_freq)):4.4}", logger=logger)
        if logger is not None:
            wandb.log({
                'Loss Field / Training': np.mean(losses_field),
                'Loss Freq / Training': np.mean(losses_freq),
                'LR': optimizer.param_groups[0]['lr'],
                'Epoch': epoch
            })

        if epoch % config.validation_frequency == 0:
            save_model(args.dir, epoch, net, optimizer, lowest_loss, "checkpoint_last")
            net.eval()
            loss = evaluate(args, config, net, valloader, logger=logger, epoch=epoch)["loss (test/val)"]
            if loss < lowest_loss:
                print_log("best model", logger=logger)
                lowest_loss = loss
                save_model(args.dir, epoch, net, optimizer, lowest_loss)
        if epoch == (config.epochs - 1):
            path = os.path.join(args.dir, "checkpoint_best")
            net.load_state_dict(torch.load(path)["model_state_dict"])
            _ = evaluate(args, config, net, valloader, logger=logger, epoch=epoch)
    return net


def evaluate(args, config, net, dataloader, logger=None, epoch=None, verbose=True):
    report_peak_error = False
    prediction, output, field_losses = _generate_preds(args, config, net, dataloader)
    results = _evaluate(prediction, output, logger, epoch, verbose, field_losses)
    return results


def _evaluate(prediction, output, logger, epoch, verbose=True, field_losses=None):
    REPORT_L1_LOSS = True
    results = {}
    losses_per_f = torch.nn.functional.mse_loss(prediction, output, reduction="none")
    prediction, output, losses_per_f = prediction.numpy(), output.numpy(), losses_per_f.numpy()
    loss = np.mean(losses_per_f)
    results.update({"losses_per_f": losses_per_f, "loss (test/val)": loss})
    if field_losses is not None:
        results.update(field_losses)
    if REPORT_L1_LOSS is True:
        results.update({"L1 Loss / (test/val)": np.mean(np.abs(prediction - output))})
    for key in results.keys():
        if key == "losses_per_f" or key == "peak_ratio":
            continue
        if verbose is True:
            print_log(f"{key} = {results[key]:4.4}", logger=logger)
        if logger is not None:
            wandb.log({key: results[key], 'Epoch': epoch})

    return results


def _generate_preds(args, config, net, dataloader):
    with torch.no_grad():
        predictions, outputs, mse_losses, l1_losses = [], [], [], []
        out_mean, out_std, field_mean, field_std = extract_mean_std(dataloader.dataset)
        out_mean, out_std, field_mean, field_std = out_mean.to(args.device), out_std.to(args.device), field_mean.to(args.device), field_std.to(
            args.device)
        for batch in dataloader:
            image, velocity_field, vel_mean_sq, condition, frequencies = (batch[field].to(args.device) for field in fields)
            prediction_field = net(image, condition, frequencies)
            mse_losses.append(F.mse_loss(prediction_field, velocity_field).detach().cpu().numpy())
            l1_losses.append(F.l1_loss(prediction_field, velocity_field).detach().cpu().numpy())
            prediction = get_mean_from_velocity_field(prediction_field, field_mean, field_std, frequencies)
            prediction.sub_(out_mean[frequencies]).div_(out_std)
            if config.max_frequency is not None:
                prediction, vel_mean_sq = prediction[:, :config.max_frequency], vel_mean_sq[:, :config.max_frequency]
            predictions.append(prediction.detach().cpu()), outputs.append(vel_mean_sq.detach().cpu())
    return torch.vstack(predictions), torch.vstack(outputs), {
        "Loss Field / (test/val)": np.mean(mse_losses),
        "L1 Loss Field / (test/val)": np.mean(l1_losses)
    }
