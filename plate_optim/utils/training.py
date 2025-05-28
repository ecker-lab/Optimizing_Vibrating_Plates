import torch, os


def save_model(savepath, epoch, model, optimizer, loss, name="checkpoint_best"):
    # deal with torch compile key changes
    state_dict = {key.replace("_orig_mod.", ""): val for key, val in model.state_dict().items()}
    optimizer_state_dict = optimizer.state_dict() if optimizer is not None else None
    torch.save({
        'epoch': epoch,
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'loss': loss
    }, os.path.join(savepath, name))
