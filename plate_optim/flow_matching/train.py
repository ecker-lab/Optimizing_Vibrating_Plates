import torch, wandb
from torchmetrics.aggregation import MeanMetric
from flow_matching.examples.image.training.train_loop import skewed_timestep_sample
from codeutils.logger import print_log
class_drop_prob = 1
use_skewed_timesteps = True  # this should be replaced
FP16 = True
gradient_clip = 100


def train_one_epoch(model, dataloader, optimizer, lr_scheduler, device, epoch, scaler, logger=None):
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)

    model.train()
    for iteration, (x1, _) in enumerate(dataloader):
        optimizer.zero_grad()
        batch_loss.reset()
        x1 = x1.to(device)
        x0 = torch.randn_like(x1).to(device)
        if use_skewed_timesteps:
            t = skewed_timestep_sample(x1.shape[0], device=device)
        else:
            t = torch.torch.rand(x1.shape[0]).to(device)
        xt = t * x1 + (1 - t) * x0
        dxdt = x1 - x0

        with torch.amp.autocast('cuda', enabled=FP16):
            prediction = model(xt, t, extra={})
            loss = torch.nn.functional.mse_loss(prediction, dxdt)

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        wandb.log({'batch_loss': loss_value})

        if not torch.isfinite(loss):
            raise ValueError(f"Loss is {loss_value}, stopping training")
        scaler.scale(loss).backward()
        if gradient_clip is not None:
            scaler.unscale_(optimizer)  # in_place unscale
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        if iteration % 20 == 0:
            print_log(f"Epoch {epoch}: [{iteration}/{len(dataloader)}]: \
                    loss = {batch_loss.compute()}, lr = {optimizer.param_groups[0]["lr"]}", logger=logger)

    lr_scheduler.step()
    wandb.log({'epoch_loss': epoch_loss.compute(), 'lr': optimizer.param_groups[0]["lr"], 'epoch': epoch})
    return epoch_loss.compute().detach().cpu().numpy()
