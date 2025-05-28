import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import ast
import torch
import wandb
import time
import json

from plate_optim.utils.data import get_moments
from plate_optim.regression.regression_model import get_net, get_mean_from_velocity_field
from plate_optim.utils.guidance import get_loss_fn, _callable_constructor, get_peak_loss_fn, get_first_peak_loss_fn
from codeutils.logger import init_train_logger, print_log
from plate_optim.utils.logging import update_best_responses
from flow_matching.solver import ODESolver
from plate_optim.flow_matching.model.unet import UNetModel
from plate_optim.metrics.manufacturing import mean_valid_pixels, check_boundary_condition, check_beading_size, check_derivative, check_beading_space
from plate_optim.flow_matching.evaluation import convert_to_beading_plate, convert_to_fm_sample
from plate_optim.pattern_generation import postprocess

import numpy as np

mean_conditional_param = [50, 0.5, 0.5] # Boundary condition, force_position x, force_position y
std_conditional_param = [28.8675, 0.173205,  0.173205] 
model_resolution = [96, 128]

parser = argparse.ArgumentParser()
parser.add_argument("--regression_path", type=str)
parser.add_argument("--flow_matching_path", type=str)
parser.add_argument("--dir", default="debug", type=str)
parser.add_argument("--run_name", default="flow_matching", type=str)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--min_freq", default=200, type=int)
parser.add_argument("--max_freq", default=300, type=int)
parser.add_argument("--n_freqs", default=101, type=int)
parser.add_argument("--move_first_peak", choices=[True, False], type=lambda x: x == 'True', default=False)
parser.add_argument("--peak_loss", choices=[True, False], type=lambda x: x == 'True', default=False)

parser.add_argument('--extra_conditions', default=[0, 0.2, 0.2], type=ast.literal_eval, help='spring, fx, fy')
parser.add_argument('--resolution', default=[121, 181], type=ast.literal_eval, help='spzzring, fx, fy')
parser.add_argument("--n_plates", default=500, type=int)
parser.add_argument("--n_candidates", default=16, type=int)
parser.add_argument("--step_size", default=0.05, type=float)
parser.add_argument("--alpha", default=0.25, type=float)
parser.add_argument("--logging", choices=[True, False], type=lambda x: x == 'True', default=False)
parser.add_argument("--return_intermediates", choices=[True, False], type=lambda x: x == 'True', default=False)
parser.add_argument("--norm_grad", choices=[True, False], type=lambda x: x == 'True', default=True)
parser.add_argument("--apply_postprocessing", choices=[True, False], type=lambda x: x == 'True', default=True)


def batch_min(tensor):
    return torch.min(tensor.view(tensor.size(0), -1), dim=-1).values
def batch_max(tensor):
    return torch.max(tensor.view(tensor.size(0), -1), dim=-1).values
def rescale(x):
    x_min, x_max = batch_min(x).view(x.size(0), *[1] * (x.dim() - 1)), batch_max(x).view(x.size(0), *[1] * (x.dim() - 1))
    return (x - x_min) / (x_max - x_min) * 0.02


def do_guided_flow(args):
    best_response = 1000
    mean_responses_series = []
    responses_timeseries = []
    if args.logging:
        logger = init_train_logger(args)
        wandb.init(project="plate_optim", name=args.run_name, config=args)
    else:
        logger = None
    _, _, field_mean, field_std = get_moments(device='cuda')

    config = json.load(open(os.path.join(os.path.dirname(args.flow_matching_path), 'model_config.json')))
    model = UNetModel(**config).cuda()
    model.load_state_dict(torch.load(args.flow_matching_path)['model_state_dict'])
    model.eval()
    print_log(f"Model loaded from {args.flow_matching_path}", logger)
    regression_model = get_net(conditional=True, len_conditional=3, scaling_factor=32).cuda()
    regression_model.load_state_dict(torch.load(args.regression_path)['model_state_dict'])
    regression_model.eval()

    frequencies = ((torch.linspace(args.min_freq, args.max_freq, args.n_freqs) / 150 - 1).unsqueeze(0).repeat(args.batch_size, 1).cuda())
    condition = (torch.tensor(args.extra_conditions) - torch.tensor(mean_conditional_param)) / torch.tensor(std_conditional_param)
    condition = condition.unsqueeze(0).repeat(args.batch_size, 1).cuda() 

    # Construct loss function based on args to call
    if not args.move_first_peak and not args.peak_loss:
        loss_fn = get_loss_fn(regression_model, condition, frequencies, field_mean, field_std)
    elif args.peak_loss:
        min_freq, max_freq, n_freqs = args.min_freq - 50, args.max_freq + 50, args.n_freqs + 100  
        frequencies_loss = torch.linspace(min_freq, max_freq, n_freqs).cuda()
        _frequencies = ((frequencies_loss / 150 - 1).unsqueeze(0).repeat(args.batch_size, 1).cuda())
        loss_fn, peak_loss_fn = get_peak_loss_fn(regression_model, condition, frequencies_loss, _frequencies, field_mean, field_std, \
                                interval=(args.min_freq, args.max_freq), sigma=2)
    elif args.move_first_peak:    
        assert args.n_freqs == 300, "PeakLoss only works with 300 frequencies"
        frequencies = ((torch.linspace(args.min_freq, args.max_freq, args.n_freqs) / 150 - 1).unsqueeze(0).repeat(args.batch_size, 1).cuda())
        loss_fn, peak_loss_fn = get_first_peak_loss_fn(regression_model, condition, frequencies, field_mean, field_std)
    else:
        raise NotImplementedError
    
    velocity_model = _callable_constructor(model, loss_fn)
    solver = ODESolver(velocity_model=velocity_model)

    candidate_imgs = torch.empty((args.n_candidates, 1, *model_resolution))
    candidate_responses = torch.ones((args.n_candidates, frequencies.shape[1])) * 10000
    candidate_intermediates = torch.empty((args.n_candidates, int(1 / args.step_size + 2), 1, *model_resolution))
    start_time = time.time()

    for i in range(0, args.n_plates // args.batch_size):
        # perform guided flow matching inference
        x_0 = torch.randn([args.batch_size, 1, *model_resolution], dtype=torch.float32, device='cuda')        
        samples = solver.sample(
            time_grid=torch.linspace(0, 1, int(1 / args.step_size) + 1),
            x_init=x_0,
            method='midpoint',
            step_size=args.step_size,
            alpha=args.alpha,
            do_grad_step=True,
            return_intermediates=args.return_intermediates,
            norm_grad=args.norm_grad,
            #estimate_x1=True,
            #pure_solver=pure_solver,
        )
        if args.return_intermediates:
            intermediates = samples # steps x batch x 1 x 96 x 128, permute steps and batch
            intermediates = intermediates.permute(1, 0, 2, 3, 4) # batch x steps x 1 x 96 x 128
            intermediates = torch.cat([x_0.unsqueeze(1), intermediates], dim=1)
            samples = samples[-1]

        # apply postprocessing
        if args.apply_postprocessing:
            samples = convert_to_beading_plate(samples.cpu(), scaling_factor=2)
            samples = postprocess(samples)
            samples = convert_to_fm_sample(samples).cuda().float()

        # get estimated response
        with torch.no_grad():
            prediction_field = regression_model(samples, condition, frequencies)
            prediction = get_mean_from_velocity_field(prediction_field, field_mean, field_std, frequencies)

        # sort candidates and stack into candidate objects
        ## compute order
        all_preds = torch.cat([candidate_responses, prediction.detach().cpu()], dim=0)
        if not args.peak_loss and not args.move_first_peak:
            loss = torch.mean(all_preds, dim=1)
        else:
            loss = peak_loss_fn(all_preds) + 0.001 * torch.mean(all_preds, dim=1)
        idx = torch.argsort(loss)

        ## reorder and update elements
        candidate_responses = all_preds[idx[:args.n_candidates]]
        candidate_imgs = torch.cat([candidate_imgs, samples.detach().cpu()], dim=0)[idx[:args.n_candidates]]
        mean_responses_series.append(torch.mean(candidate_responses, dim=1))
        if args.return_intermediates:
            candidate_intermediates = torch.cat([candidate_intermediates, intermediates.detach().cpu()], dim=0)[idx[:args.n_candidates]]
        print_log(f"Batch {i} done", logger)
        print_log(f"mean candidate_responses: {torch.mean(candidate_responses, dim=1)}", logger)
        if args.logging:
            wandb.log({"candidate_imgs": wandb.Image(candidate_imgs), "batch": i, 
                        "mean_responses_model": torch.mean(candidate_responses), \
                        'mean_responses_grouped': mean_responses_series[-1],\
                        "nfe_counter": regression_model.nfe_counter})
            
        best_response, responses_timeseries = update_best_responses(candidate_responses, \
                                                            responses_timeseries, best_response, regression_model.nfe_counter, start_time)
    candidate_imgs_ = convert_to_beading_plate(candidate_imgs, scaling_factor=2)
    candidate_imgs = convert_to_beading_plate(candidate_imgs)

    
    end_time = time.time()
    nfe_counter = regression_model.nfe_counter
    mean_valid_pixels_value = mean_valid_pixels(candidate_imgs_.numpy())

    if args.logging:
        wandb.run.summary["mean_model_response"] = torch.mean(candidate_responses).item()
        wandb.run.summary["lowest_model_response"] = torch.mean(candidate_responses, dim=1).min().item()
        wandb.run.summary["nfe_counter"] = nfe_counter
        wandb.run.summary["mean_valid_pixels"] = mean_valid_pixels_value

    print_log(f'nfe_counter: {(nfe_counter)}', logger)
    print_log(f'Solver time: {end_time - start_time:.2f} seconds', logger)
    print_log(f'Mean valid pixels: {mean_valid_pixels_value}', logger)


    space_mask = np.mean([check_beading_space(img) for img in candidate_imgs.numpy()])
    size_mask =  np.mean([check_beading_size(img) for img in candidate_imgs.numpy()])
    derivative_mask =  np.mean([check_derivative(img) for img in candidate_imgs.numpy()])
    bc_mask =  np.mean([check_boundary_condition(img) for img in candidate_imgs.numpy()])
    print("space, size, derivative, bc:", np.mean(space_mask), np.mean(size_mask), np.mean(derivative_mask), np.mean(bc_mask))

    print_log(f"candidate_responses: {torch.mean(candidate_responses, dim=1)}", logger)
    print_log(f"candidate_responses: {torch.mean(candidate_responses)}", logger)

    # save candidates
    os.makedirs(args.dir, exist_ok=True)
    torch.save(candidate_imgs, os.path.join(args.dir, "candidate_plates.pt"))
    torch.save(candidate_responses, os.path.join(args.dir, "candidate_responses.pt"))
    frequencies = torch.linspace(args.min_freq, args.max_freq, args.n_freqs)
    torch.save(frequencies, os.path.join(args.dir, "frequencies.pt"))
    torch.save(args.extra_conditions, os.path.join(args.dir, "physical_parameters.pt"))
    torch.save(candidate_intermediates, os.path.join(args.dir, "candidate_intermediates.pt"))
    torch.save(torch.tensor(responses_timeseries), os.path.join(args.dir, "responses_timeseries.pt"))
    torch.save(torch.stack(mean_responses_series, dim=0), os.path.join(args.dir, "mean_responses_series.pt"))
    torch.save(mean_valid_pixels_value, os.path.join(args.dir, "mean_valid_pixels_value.pt"))
               
    # save wandb run id to match model predictions to fem results
    if args.logging:
        with open(os.path.join(args.dir, "wandb_run_id.txt"), "w") as f:
            f.write(str(wandb.run.id))
        return wandb.run.id


if __name__ == "__main__":
    args = parser.parse_args()
    do_guided_flow(args)