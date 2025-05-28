import sys

from scipy.optimize import differential_evolution
from plate_optim.utils.data import get_moments
from plate_optim.pattern_generation import ParametricBeading, BeadingTransition, apply_length_scale_constraint
from plate_optim.regression.regression_model import get_net, get_mean_from_velocity_field
from plate_optim.utils.callables_fem import plate_prediction_multiprocessing
import argparse, os, torch
import numpy as np
import ast
import time


mean_conditional_param = [50, 0.5, 0.5] # Boundary condition, force_position x, force_position y
std_conditional_param = [28.8675, 0.173205,  0.173205] 
out_mean, out_std, field_mean, field_std = get_moments(device='cuda')

count = 0
save_dir = None
iter_hist = []


def get_beadings_scaled(theta, para_beading, user_args):
    beadings = para_beading.draw(theta, user_args.max_n_lines, user_args.max_n_arcs, user_args.max_n_quads)
    beadings = (beadings / 0.02 * 2) - 1        # scale to -1 , 1
    beadings = torch.from_numpy(beadings).float().cuda()
    return beadings


def get_prediction(regression_model, beadings, frequencies, extra_conditions):
    condition = (torch.tensor(extra_conditions) - torch.tensor(mean_conditional_param)) / torch.tensor(std_conditional_param)
    condition = condition.unsqueeze(0).repeat(beadings.shape[0], 1).cuda()

    with torch.no_grad():
        prediction_field = regression_model(beadings, condition, frequencies)
        prediction = get_mean_from_velocity_field(prediction_field, field_mean, field_std, frequencies)
    return prediction


def callback(intermediate_result):
    global iter_hist
    global start_time
    iter_hist.append([intermediate_result.fun*init_mean_frf[0], count, time.time()-start_time])
    os.makedirs(save_dir, exist_ok=True)

    torch.save(torch.tensor(iter_hist), os.path.join(args.dir, "responses_timeseries.pt"))
    # Store the current best objective value
    print(f"fun evals = {count}. Obj fun value = {intermediate_result.fun*init_mean_frf[0]}, at time = {time.time()-start_time}")


def obj_fun(theta, *opt_args):
    global count, save_dir
    #print(f"ObjFun eval = {count}")
    regression_model = opt_args[0]
    para_beading = opt_args[1]
    args = opt_args[2]
    save_dir = args.dir
    init_mean_frf = opt_args[3]
    if theta.ndim == 1: theta = theta.reshape(-1,1)
    querry_size = theta.shape[1]
    count += querry_size
    print(querry_size)
    mean_frf_norm_vec = np.array([])

    for i in range(int(querry_size / args.batch_size)+1):
        theta_batch = theta[:,i*args.batch_size:(i+1)*args.batch_size]
        batch_size = theta_batch.shape[1]
        # Draw beading_pattern
        beadings = get_beadings_scaled(theta_batch, para_beading, args)
        frequencies = ((torch.linspace(args.min_freq, args.max_freq, args.n_freqs) / 150 - 1).unsqueeze(0).repeat(batch_size, 1).cuda())

        # Compute prediction
        prediction = get_prediction(regression_model, beadings, frequencies, args.extra_conditions)
        
        # Calc loss
        mean_frf = prediction.mean(1).detach().cpu().numpy()
        mean_frf_norm = mean_frf/init_mean_frf[0]
        mean_frf_norm_vec = np.concatenate((mean_frf_norm_vec, mean_frf_norm))
    # print(f"loss = {mean_frf_norm}")

        opt_args[4].append(mean_frf)

    return mean_frf_norm_vec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="debug", type=str)
    parser.add_argument("--regression_path", type=str)
    parser.add_argument("--min_freq", default=1, type=int)
    parser.add_argument("--max_freq", default=300, type=int)
    parser.add_argument("--n_freqs", default=300, type=int)
    parser.add_argument("--n_candidates", default=4, type=int)

    parser.add_argument('--extra_conditions', default=[0, 0.2, 0.2], type=ast.literal_eval, help='spring, fx, fy')
    parser.add_argument("--max_n_arcs", default=0, type=int)
    parser.add_argument("--max_n_lines", default=2, type=int)
    parser.add_argument("--max_n_quads", default=2, type=int)
    parser.add_argument("--n_pop", default=10, type=int)
    parser.add_argument("--n_max_iter", default=5, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--vectorized", choices=[True, False], type=lambda x: x == 'True', default=True)
    parser.add_argument("--comp_FEM", choices=[True, False], type=lambda x: x == 'True', default=False)
    args = parser.parse_args()

    # Get the model
    regression_model = get_net(conditional=True, len_conditional=3, scaling_factor=32).cuda()
    regression_model.load_state_dict(torch.load(args.regression_path)['model_state_dict'])
    regression_model.eval()

    # Get initial response without beading
    no_beading = torch.ones((1,1,121,181)).cuda().float()*-1
    frequencies = ((torch.linspace(args.min_freq, args.max_freq, args.n_freqs) / 150 - 1).unsqueeze(0).cuda())
    prediction = get_prediction(regression_model, no_beading, frequencies, args.extra_conditions)
    global init_mean_frf
    init_mean_frf = prediction.mean(1).detach().cpu().numpy()

    # Get the parametric beading generator
    resolution = np.array([181, 121])
    dimension = np.array([0.9, 0.6])
    n_para = args.max_n_lines * 5 + args.max_n_arcs * 8 + args.max_n_quads * 7 + 3
    eng_beading = BeadingTransition(h_bead=0.02, r_f=0.0095, r_h=0.0095, alpha_F=70 * np.pi / 180)
    para_beading = ParametricBeading(resolution=resolution, dimension=dimension, eng_beading=eng_beading, min_length_scale=0.025)

    # Genetic Opt
    call_hist = []
    bounds = [(0.0, 1.0)] * n_para

    global start_time
    start_time = time.time()
    result = differential_evolution(obj_fun, bounds=bounds, 
                                    args=(regression_model, para_beading, args, init_mean_frf, call_hist), 
                                    maxiter=args.n_max_iter, popsize=args.n_pop, disp=True, polish=False, 
                                    callback=callback, vectorized=args.vectorized)    
    elapsed_ga = time.time() - start_time

    # Optimal results
    best_idx = result.population_energies.argsort()
    best_theta = result.population[best_idx[:args.n_candidates]]
    print(result.population_energies[best_idx[:args.n_candidates]])

    beading_patterns = torch.stack([get_beadings_scaled(theta.reshape(-1, 1), para_beading, args) for theta in best_theta])[:,0]
    beading_patterns_ = torch.nn.functional.interpolate(beading_patterns, size=(96, 128), mode='bilinear', align_corners=True)
    target_freqs = (torch.linspace(args.min_freq, args.max_freq, args.n_freqs)/ 150 - 1)
    target_freqs = target_freqs.unsqueeze(0).repeat(args.n_candidates, 1).cuda()
    prediction = get_prediction(regression_model, beading_patterns_, target_freqs, args.extra_conditions)

    candidate_plates = (beading_patterns.detach().cpu() + 1) / 2 * 0.02
    candidate_responses = prediction.detach().cpu()
    print(f"candidate_responses: {torch.mean(candidate_responses, dim=1)}")

    # Save
    print("Saving...")
    os.makedirs(args.dir, exist_ok=True)
    torch.save(args.extra_conditions, os.path.join(args.dir, "physical_parameters.pt"))
    torch.save(candidate_plates.cpu().numpy(), os.path.join(args.dir, "candidate_plates.pt"))
    torch.save(candidate_responses.cpu().numpy(), os.path.join(args.dir, "candidate_responses.pt"))
    target_freqs = torch.linspace(args.min_freq, args.max_freq, args.n_freqs)
    torch.save(target_freqs, os.path.join(args.dir, "frequencies.pt"))
    torch.save(elapsed_ga, os.path.join(args.dir, "timeing.pt"))


