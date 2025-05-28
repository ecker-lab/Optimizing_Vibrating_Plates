import numpy as np
import time

def update_best_responses(candidate_responses, responses_timeseries, best_response, iter_step, start_time):
    candidate_response = candidate_responses.mean(dim=1)[0]
    time_step = time.time() - start_time
    if candidate_response < best_response:
        best_response = candidate_response
        responses_timeseries.append([best_response, iter_step, time_step])
        print('best response update at time step', time_step)
    # also append if it does not improve to have the final step and time
    if candidate_response <= best_response:
        best_response = candidate_response
        responses_timeseries.append([best_response, iter_step, time_step])
    return best_response, responses_timeseries
