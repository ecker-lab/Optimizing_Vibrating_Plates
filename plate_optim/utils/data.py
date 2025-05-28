import h5py, torch, hdf5plugin, os
from torchinterp1d import interp1d
from plate_optim.project_directories import data_dir

def get_moments(
        moments_dict_path=None,
        eps=1e-12, 
        device='cpu'
        ):
    if moments_dict_path is None:
        moments_dict_path = os.path.join(data_dir, 'D50000/moments_d50k.pt')
        if not os.path.exists(moments_dict_path):
            raise FileNotFoundError(f"The moments file was not found at the expected path: {moments_dict_path}. "
                                    "Please provide the correct path to the file.")
    moments_dict = torch.load(moments_dict_path)
    out_mean = moments_dict['out_mean']
    out_std = moments_dict['out_std']
    field_mean = moments_dict['field_mean']
    field_std = moments_dict['field_std']
    print(out_mean.shape, out_std.shape, field_mean.shape, field_std.shape)
    field_mean = convert_1d_to_interpolator(field_mean.flatten(), -1, 1)
    out_mean = convert_1d_to_interpolator(out_mean, -1, 1)
    return out_mean.to(device), out_std.to(device), field_mean.to(device), field_std.to(device)


class convert_1d_to_interpolator:
    def __init__(self, array, min_val, max_val):
        self.array = array
        self.min_val = min_val
        self.max_val = max_val
        self.x = torch.linspace(min_val, max_val, steps=array.shape[0], device=array.device)

    def __getitem__(self, xnew):
        if not isinstance(xnew, torch.Tensor):
            xnew = torch.tensor(xnew, dtype=torch.float32, device=self.array.device)
        original_shape = xnew.shape
        xnew_flat = xnew.flatten()
        interpolated_values_flat = interp1d(self.x, self.array, xnew_flat, None)
        return interpolated_values_flat.view(original_shape)

    def cuda(self):
        self.array = self.array.cuda()
        self.x = self.x.cuda()
        return self

    def to(self, device, dtype=None):
        self.array = self.array.to(device, dtype=dtype)
        self.x = self.x.to(device, dtype=dtype)
        return self


def compute_and_save_moments(
        reference_path, 
        save_path,
        eps=1e-12, 
        ):

    with h5py.File(reference_path, 'r') as f:
        data = torch.from_numpy(f["z_vel_abs"][:]).float()
        data = torch.log(torch.square(data) + eps)
        field_mean = torch.mean(data, axis=(0, 2, 3))
        field_std = torch.std(data)
        data = torch.from_numpy(f["z_vel_mean_sq"][:])
        out_mean = torch.mean(data, dim=0).float()
        out_std = torch.std((data - out_mean)).float()

    moments_dict = {
        'out_mean': out_mean,
        'out_std': out_std,
        'field_mean': field_mean,
        'field_std': field_std
    }
    torch.save(moments_dict, save_path)