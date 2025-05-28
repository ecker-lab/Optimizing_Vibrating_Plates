import torch
from plate_optim.regression.regression_model import get_mean_from_velocity_field
from plate_optim.utils.softlocalmax import PeakMoverLoss, PositionIntervalLoss


def get_loss_fn(regression_model, condition_, frequencies, field_mean, field_std):
    def loss_fn(x, condition=condition_, frequencies=frequencies):
        pred = regression_model(x, condition, frequencies)
        pred = get_mean_from_velocity_field(pred, field_mean, field_std, frequencies)
        return pred.mean()
    return loss_fn


def get_first_peak_loss_fn(regression_model, condition_, frequencies, field_mean, field_std):
    peak_loss = PeakMoverLoss() 
    def loss_fn(x, condition=condition_, frequencies=frequencies):
        pred = regression_model(x, condition, frequencies)
        pred = get_mean_from_velocity_field(pred, field_mean, field_std, frequencies)
        loss = peak_loss(pred)
        return loss
    return loss_fn, peak_loss


def get_peak_loss_fn(regression_model, condition_, frequencies_loss, frequencies, field_mean, field_std, interval, sigma=2):
    peak_loss = PositionIntervalLoss(interval, freqs=frequencies_loss, sigma=sigma) 
    def loss_fn(x, condition=condition_, frequencies=frequencies):
        pred = regression_model(x, condition, frequencies)
        pred = get_mean_from_velocity_field(pred, field_mean, field_std, frequencies)
        loss = peak_loss(pred)
        return loss
    return loss_fn, peak_loss


def _callable_constructor(flow_matching_model, regression_loss_model, save_path=None):
    def cosine_scheduler(t, max_lr, min_lr):
        return (0.5 * (1 + torch.cos(torch.pi * t)) * (max_lr - min_lr) + min_lr).float().cuda()

    def _callable(x, t, alpha=0.5, do_grad_step=True, norm_grad=True, **kwargs):
        do_grad_step = do_grad_step and t < 0.75
        do_flow_step = t >= 0
        t = torch.zeros(x.shape[0], device=x.device) + t.to(x.device)
        if do_flow_step:
            v_flow = flow_matching_model(x, t, {})
        else :
            v_flow = torch.zeros_like(x)
        if do_grad_step:
            with torch.enable_grad(): # , torch.amp.autocast('cuda') that might not work with torch.autograd.grad
                x = x.detach().clone().requires_grad_(True)
                loss = regression_loss_model(x).mean()
                grad = torch.autograd.grad(loss, x)[0]

            # set gradient to zero for all pixels at boundary. boundary defined as 7 pixels from the edge
            grad[:, :, :7, :] = 0
            grad[:, :, -7:, :] = 0
            grad[:, :, :, :7] = 0
            grad[:, :, :, -7:] = 0
            
            velocity_norm = torch.linalg.norm(v_flow.view(x.shape[0], -1), dim=1).view(x.shape[0], 1, 1, 1)
            grad_norm = torch.linalg.norm(grad.view(x.shape[0], -1), dim=1).view(x.shape[0], 1, 1, 1)

            if norm_grad:
                grad = grad / grad_norm * velocity_norm 
            else:
                print(f"v_norm: {velocity_norm.mean().item():.3f}, gradient_norm: {grad_norm.mean().item():.3f}")
            print(f"[t={t[0]:.3f}] loss: {loss.item():.3f}")
        else:
            grad = torch.zeros_like(x)
        # save velocity field and grad
        if save_path is not None:
            import os
            os.makedirs(save_path, exist_ok=True)
            torch.save(v_flow.cpu(), os.path.join(save_path, f"v_flow_{t[0]:.3f}.pt"))
            torch.save(grad.cpu(), os.path.join(save_path, f"grad_{t[0]:.3f}.pt"))
        dxdt = v_flow - alpha * grad * cosine_scheduler(t, 1, 1e-1).view(x.shape[0], 1, 1, 1)
        return dxdt
    return _callable

