# %%
import wandb
import torch
import matplotlib.pyplot as plt
from plate_optim.flow_matching.train import train_one_epoch, FP16
from plate_optim.flow_matching.evaluation import solve_for_samples, make_fig, convert_to_beading_plate, VelocityModel, compute_log_likelihood
from plate_optim.utils.training import save_model
from codeutils.logger import init_train_logger, print_log
from plate_optim.metrics.manufacturing import calc_beading_ratio, mean_valid_pixels


def train(args, model, dataloader, optimizer, lr_scheduler, logger=None, validation_dataloader=None):
    scaler = torch.amp.GradScaler(device='cuda', enabled=FP16)
    torch.set_float32_matmul_precision("high")
    if args.compile:
        _model = torch.compile(model)
    else:
        _model = model

    # Get validation samples for likelihood computation
    if validation_dataloader is not None:
        val_samples = next(iter(validation_dataloader))[0].cuda()
    else:
        print_log("No validation dataloader provided, using samples from training set", logger)
        val_samples = next(iter(dataloader))[0][:8].cuda()
    
    # train
    lowest_loss = 1e9
    for epoch in range(args.epochs):
        loss = train_one_epoch(_model, dataloader, optimizer, lr_scheduler, torch.device('cuda'), epoch, scaler, logger=logger)
        print_log(f"Epoch {epoch}: loss = {loss}", logger)
        
        # Generate samples and compute metrics
        samples = solve_for_samples(model, resolution=args.resolution)
        plates = convert_to_beading_plate(samples).cpu().numpy()
        manufacturability_metric = mean_valid_pixels(plates)
        avg_likelihood = compute_log_likelihood(model, val_samples)
        print_log(f"Epoch {epoch}: avg log likelihood = {avg_likelihood}", logger)
            
        wandb.log({
            "generated_samples": wandb.Image(make_fig(samples)), 
            'manufacturability_metric': manufacturability_metric, 
            "epoch": epoch,
            "avg_log_likelihood": avg_likelihood,
            "loss": loss,
        })
        
        plt.close('all')
        if loss < lowest_loss:
            save_model(args.dir, epoch, model, optimizer, loss)
            lowest_loss = loss
        save_model(args.dir, epoch, model, None, loss, name=f"checkpoint_{epoch}")

    solve_for_samples(model, resolution=args.resolution)
