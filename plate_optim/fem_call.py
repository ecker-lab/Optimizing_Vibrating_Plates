"""
This file does not contain a full implementation of numerical simulation. The full implementation requires an external finite element solver.
The function plate_prediction_multiprocessing can be changed to call a finite element solver if one is available.
"""

import argparse
import torch
import wandb
import os
import numpy as np
from codeutils.logger import init_train_logger, print_log
from concurrent.futures import ProcessPoolExecutor
import sys


def plate_prediction_multiprocessing(freqs, *kwargs):
    # this function is not implemented
    frequency_responses = np.zeros(len(freqs))
    return frequency_responses, None


def suppress_output(func, *args, **kwargs):
    result = func(*args, **kwargs)
    return result


def process_sample(i, img, freqs, n_processes_fem, phy_para, logger):
    print_log(f"Sample {i}", logger=logger)
    return suppress_output(plate_prediction_multiprocessing, freqs=freqs, n_processes=n_processes_fem, beading_pattern=img, phy_para=phy_para)[0]


def run_fem_for_directory(path, logger=None, n_processes_fem=16, n_processes=4, return_results=False):
    images = torch.load(path + "/candidate_plates.pt")
    frequencies = torch.load(path + "/frequencies.pt")
    phy_para = torch.load(path + "/physical_parameters.pt")
    print_log(f"phy_para: {phy_para}", logger)
    print_log(f'freqs: {frequencies.shape}', logger)
    args_list = [(i, images[i], frequencies, n_processes_fem, phy_para, logger) for i in range(len(images))]

    frequency_responses = []
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        futures = [executor.submit(process_sample, *args) for args in args_list]
        for future in futures:
            frequency_responses.append(future.result())
    
    torch.save(frequency_responses, path + "/fem_solutions.pt")
    if return_results:
        return np.array(frequency_responses)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default='debug', type=str)
    parser.add_argument("--logging", choices=[True, False], type=lambda x: x == 'True', default=True)
    args = parser.parse_args()
    print(args)
    logger = None
    if args.logging:
        logger = init_train_logger(args, args)
    frequency_responses = run_fem_for_directory(args.dir, logger, return_results=True)

    print_log(f"Mean responses: {frequency_responses.mean(axis=1)}", logger)
    print_log(f"Mean responses: {np.mean(frequency_responses).item():2.2f} ({np.std(frequency_responses.mean(axis=1)):2.2f}), {frequency_responses.mean(axis=1).min().item():2.2f}", logger)