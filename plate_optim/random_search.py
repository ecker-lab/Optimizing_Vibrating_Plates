import argparse
import os

import ast
import torch
import wandb
import time
from plate_optim.metrics.manufacturing import mean_valid_pixels 

from plate_optim.pattern_generation import StandardPlateDataset
from plate_optim.utils.data import get_moments
from plate_optim.regression.regression_model import get_net, get_mean_from_velocity_field
from plate_optim.utils.logging import update_best_responses
from torchvision import transforms
from codeutils.logger import init_train_logger, print_log
mean_conditional_param = [50, 0.5, 0.5] # Boundary condition, force_position x, force_position y
std_conditional_param = [28.8675, 0.173205,  0.173205] 


parser = argparse.ArgumentParser()
parser.add_argument("--regression_path", type=str)
parser.add_argument("--dir", default="debug", type=str)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--min_freq", default=100, type=int)
parser.add_argument("--max_freq", default=200, type=int)
parser.add_argument("--n_freqs", default=101, type=int)
parser.add_argument("--name", default='random_search', type=str)

parser.add_argument('--extra_conditions', default=[0, 0.2, 0.2], type=ast.literal_eval, help='spring, fx, fy')
parser.add_argument('--resolution', default=[121, 181], type=ast.literal_eval, help='spring, fx, fy')
parser.add_argument("--n_plates", default=10000, type=int)
parser.add_argument("--n_candidates", default=16, type=int)
parser.add_argument("--logging", choices=[True, False], type=lambda x: x == 'True', default=False)


def batch_min(tensor):
    return torch.min(tensor.view(tensor.size(0), -1), dim=-1).values
def batch_max(tensor):
    return torch.max(tensor.view(tensor.size(0), -1), dim=-1).values
def rescale(x):
    x_min, x_max = batch_min(x).view(x.size(0), *[1] * (x.dim() - 1)), batch_max(x).view(x.size(0), *[1] * (x.dim() - 1))
    return (x - x_min) / (x_max - x_min) * 0.02

def normalize(x):
    return x / 0.02 * 2.0 - 1.0

def do_brute_force(args):
    best_response = 1000
    mean_responses_series = []
    responses_timeseries = []
    if args.logging:
        logger = init_train_logger(args)
        wandb.init(project="plate_optim", name=args.name, config=args)
    else:
        logger = None
    # define required objects
    out_mean, out_std, field_mean, field_std = get_moments(device='cuda')
    regression_model = get_net(conditional=True, len_conditional=3, scaling_factor=32).cuda()
    regression_model.load_state_dict(torch.load(args.regression_path)['model_state_dict'])
    regression_model.eval()
    frequencies = ((torch.linspace(args.min_freq, args.max_freq, args.n_freqs) / 150 - 1).unsqueeze(0).repeat(args.batch_size, 1).cuda())
    
    condition = (torch.tensor(args.extra_conditions) - torch.tensor(mean_conditional_param)) / torch.tensor(std_conditional_param)
    condition = condition.unsqueeze(0).repeat(args.batch_size, 1).cuda() 
    

    
    transform = transforms.Compose([transforms.Resize(args.resolution), normalize])
    start_time = time.time()
    dataset = StandardPlateDataset(transform=transform, size=args.n_plates)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    candidate_imgs = torch.empty((args.n_candidates, 1, 121, 181))
    candidate_responses = torch.ones((args.n_candidates, args.n_freqs)) * 1000

    for data_iter_step, (samples, labels) in enumerate(dataloader):
        with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
            prediction_field = regression_model(samples.cuda(), condition, frequencies)
            prediction = get_mean_from_velocity_field(prediction_field, field_mean, field_std, frequencies)

        # sort candidates and stack into candidate objects
        all_preds = torch.cat([candidate_responses, prediction.detach().cpu()], dim=0)
        idx = torch.argsort(torch.mean(all_preds, dim=1))
        candidate_responses = all_preds[idx[:args.n_candidates]]
        candidate_imgs = torch.cat([candidate_imgs, samples], dim=0)[idx[:args.n_candidates]]

        # update timeseries
        best_response, responses_timeseries = update_best_responses(candidate_responses, \
                                                                   responses_timeseries, best_response, regression_model.nfe_counter, start_time)
        mean_responses_series.append(torch.mean(candidate_responses, dim=1))
        if data_iter_step % 50 == 0:
            print_log(f"Batch {data_iter_step} done", logger)
            print_log(f"mean candidate_responses: {torch.mean(candidate_responses, dim=1)}", logger)
            if args.logging:
                wandb.log({"candidate_imgs": wandb.Image(candidate_imgs), "batch": data_iter_step, 
                       "mean_responses_model": torch.mean(candidate_responses), 'mean_responses_grouped': mean_responses_series[-1],\
                       "nfe_counter": regression_model.nfe_counter})
    end_time = time.time()
    nfe_counter = regression_model.nfe_counter
    # rescale beading patterns imgs to [0, 0.02]
    candidate_imgs = rescale(candidate_imgs)
    mean_valid_pixels_value = mean_valid_pixels(candidate_imgs.numpy()[:,0])

    if args.logging:
        wandb.run.summary["mean_model_response"] = torch.mean(candidate_responses).item()
        wandb.run.summary["lowest_model_response"] = torch.mean(candidate_responses, dim=1).min().item()
        wandb.run.summary["nfe_counter"] = nfe_counter
        wandb.run.summary["mean_valid_pixels"] = mean_valid_pixels_value
    # save candidates
    os.makedirs(args.dir, exist_ok=True)
    torch.save(candidate_imgs, os.path.join(args.dir, "candidate_plates.pt"))
    torch.save(candidate_responses, os.path.join(args.dir, "candidate_responses.pt"))
    frequencies = torch.linspace(args.min_freq, args.max_freq, args.n_freqs)
    torch.save(frequencies, os.path.join(args.dir, "frequencies.pt"))
    torch.save(args.extra_conditions, os.path.join(args.dir, "physical_parameters.pt"))
    torch.save(torch.tensor(responses_timeseries), os.path.join(args.dir, "responses_timeseries.pt"))
    torch.save(torch.stack(mean_responses_series, dim=0), os.path.join(args.dir, "mean_responses_series.pt"))

    print_log(f'nfe_counter: {(nfe_counter)}', logger)
    print_log(f'Solver time: {end_time - start_time:.2f} seconds', logger)
    print_log(f'Mean valid pixels: {mean_valid_pixels_value}', logger)
    print_log(f"candidate_responses: {torch.mean(candidate_responses, dim=1)}", logger)

    # save wandb run id to match model predictions to fem results
    if args.logging:
        with open(os.path.join(args.dir, "wandb_run_id.txt"), "w") as f:
            f.write(str(wandb.run.id))
        return wandb.run.id


if __name__ == "__main__":
    args = parser.parse_args()
    do_brute_force(args)