import torch
from flow_matching.solver import ODESolver
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np


class VelocityModel:
    def __init__(self, model):
        self.nfe_counter = 0
        self.model = model

    def __call__(self, x: torch.Tensor, t: torch.Tensor, grad_enabled=False):
        t = torch.zeros(x.shape[0], device=x.device) + t
        with torch.amp.autocast('cuda'), torch.set_grad_enabled(grad_enabled):
            result = self.model(x, t)        
        self.nfe_counter += 1
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter
    

class AutoGuidanceModel:
    def __init__(self, model1, model2):
        self.nfe_counter = 0
        self.model1 = model1
        self.model2 = model2

    def __call__(self, x: torch.Tensor, t: torch.Tensor, label={}, autoguidance_scale: float = 0.0, grad_enabled=False) -> torch.Tensor:
        assert autoguidance_scale >= 0.0 and autoguidance_scale <= 1.0
        t = torch.zeros(x.shape[0], device=x.device) + t
        with torch.amp.autocast('cuda'), torch.set_grad_enabled(grad_enabled):
            out_model1 = self.model1(x, t, extra=label)
            if autoguidance_scale > 0.0:
                out_model2 = self.model2(x, t, extra=label)
            else:
                out_model2 = torch.zeros_like(out_model1)
        result = (1.0 + autoguidance_scale) * out_model1 - autoguidance_scale * out_model2
        
        self.nfe_counter += 1
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def solve_for_samples(model, n_samples=10, resolution=(28, 28), device='cuda', seed=42):
    model.eval()
    model = VelocityModel(model=model)   
    if seed is not None:
        torch.manual_seed(seed)
    x_0 = torch.randn([n_samples, 1, *resolution], dtype=torch.float32, device=device)    
    solver = ODESolver(velocity_model=model)
    samples = solver.sample(
        x_init=x_0,
        method='midpoint',
        step_size=0.05,
    )
    return samples


def make_fig(samples):
    n_samples = samples.shape[0]
    #print(samples.view(n_samples, -1).min(dim=(1))[0], samples.view(n_samples, -1).max(dim=1)[0])
    fig, axes = plt.subplots(2, 4, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
            ax.imshow(samples[i].cpu().numpy().squeeze(), cmap='gray', vmin=-1, vmax=1)
            ax.axis('off')
    return fig


def convert_to_beading_plate(samples, scaling_factor=1, resolution=np.array([121, 181])):
    def batch_min(tensor):
        return torch.min(tensor.view(tensor.size(0), -1), dim=-1).values
    def batch_max(tensor):
        return torch.max(tensor.view(tensor.size(0), -1), dim=-1).values
    def rescale(x):
        x_min, x_max = batch_min(x).view(x.size(0), *[1] * (x.dim() - 1)), batch_max(x).view(x.size(0), *[1] * (x.dim() - 1))
        return (x - x_min) / (x_max - x_min) * 0.02
    resolution = (resolution*scaling_factor).astype(int).tolist()
    samples = torch.nn.functional.interpolate(samples, size=resolution, mode='bilinear', align_corners=True)
    samples = torch.clamp(samples, -1, 1)
    # rescale beading patterns imgs to [0, 0.02]
    samples = rescale(samples)
    # set pixels to zero based on threshold
    samples[samples < (0.02 / 100)] = 0
    # set pixels at boundary to 0
    pad_h, pad_w = 7, 7
    pad_h = int(pad_h * scaling_factor)
    pad_w = int(pad_w * scaling_factor)
    samples[:,:, :pad_h]=0
    samples[:,:, -pad_h:]=0
    samples[:,:,:, :pad_w]=0
    samples[:,:,:, -pad_w:]=0

    return samples.squeeze(1)



def convert_to_fm_sample(beading_plate, mean=0.0051512644, std=0.00851558, resolution=(96, 128)):    
    transform = transforms.Resize(resolution)
    beading_plate = transform(beading_plate)
    samples = beading_plate / 0.02 * 2 - 1 # rescale to [-1, 1]

    return samples


def log_p0(x):
    """Log probability of the prior distribution (standard normal)"""
    return torch.distributions.Normal(0, 1).log_prob(x).sum(dim=(1, 2, 3))


def compute_log_likelihood(model, val_samples, num_repetitions=2):
    """Compute average log likelihood for validation samples.
    
    Args:
        model: The flow matching model
        val_samples: Tensor of validation samples [B, C, H, W]
        num_repetitions: Number of times to repeat likelihood computation for averaging
        
    Returns:
        float: Average log likelihood across validation samples
    """
    model.eval()
    velocity_model = VelocityModel(model=model)
    solver = ODESolver(velocity_model=velocity_model)
    
    total_likelihood = torch.zeros(val_samples.shape[0], device=val_samples.device)
    
    with torch.no_grad():
        for i in range(num_repetitions):
            _, likelihood = solver.compute_likelihood(
                x_1=val_samples,
                log_p0=log_p0,
                method='midpoint',
                step_size=0.05,
                exact_divergence=False,
                grad_enabled=True
            )
            total_likelihood += likelihood
            
    avg_likelihood = total_likelihood / num_repetitions
    return avg_likelihood.mean().item()