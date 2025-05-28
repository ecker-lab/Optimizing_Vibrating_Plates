import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from codeutils.logger import print_log
import h5py
import hdf5plugin
import os
import torch.nn.functional as F
from plate_optim.project_directories import data_dir
from plate_optim.utils.data import convert_1d_to_interpolator, get_moments
import csv


class HDF5Dataset(Dataset):
    def __init__(self, config, data_paths, test=True, normalization=True, sample_idx=slice(None), freq_idx=None):
        self.data_paths = [os.path.join(data_dir, data_path) for data_path in data_paths]
        self.keys = set(config.dataset_keys.copy())
        self.files = [(h5py.File(path, 'r'), path) for path in self.data_paths]
        self.normalization = normalization
        self.freq_sampling = test == False and config.freq_sample == True
        self.freq_sampling_limit = config.freq_limit if hasattr(config, "freq_limit") else 300
        self.config = config
        self.files_loading = [{key: torch.from_numpy(f[key][:].astype(np.float32)) for key in self.keys} for f, path in self.files]
        for f, path in self.files:
            f.close()
        del self.files  
        self.data_augmentation = test == False
        
        if hasattr(config, "freq_limit") and freq_idx is None and test == False:
            n_samples, n_freqs = self.files["z_vel_mean_sq"].shape[:2]
            print('sampling new freqs')
            freq_idx = torch.stack([torch.randperm(n_freqs)[:config.freq_limit] for _ in range(n_samples)]).numpy()
        if freq_idx is None:
            self.files = {key: torch.cat([d[key] for d in self.files_loading], dim=0)[sample_idx] for key in self.keys}
        else:
            self.files = {}
            for key in {"bead_patterns", "phy_para"}:
                self.files[key] = torch.cat([d[key] for d in self.files_loading], dim=0)[sample_idx]
            sample_idx = np.repeat(sample_idx.reshape(-1, 1), freq_idx.shape[1], axis=1)
            for key in self.keys - {"bead_patterns", "phy_para"}:
                self.files[key] = torch.cat([d[key] for d in self.files_loading], dim=0)[sample_idx, freq_idx]
            del self.files_loading

        if self.normalization is True:
            self.files["bead_patterns"] = self.files["bead_patterns"].unsqueeze(1) / self.files["bead_patterns"].max()
            self.files["bead_patterns"] = self.files["bead_patterns"] * 2 - 1  # scale to [-1, 1]
            print_log(f'normalize with {config.data_path_ref}', logger=None)
            out_mean, out_std, field_mean, field_std = get_moments(os.path.join(data_dir, config.data_path_ref), device='cpu')
            self.out_mean, self.out_std, self.field_mean, self.field_std = out_mean, out_std, field_mean, field_std
            self.normalize_frequencies()
            self.normalize_frequency_response()
            if "z_vel_abs" in self.keys:
                self.normalize_field_solution()
            self.handle_conditional_params(config)

    def handle_conditional_params(self, config):
        self.files["phy_para"] = (self.files["phy_para"] - torch.tensor(config.mean_conditional_param).float().unsqueeze(0))\
        / torch.tensor(config.std_conditional_param).float().unsqueeze(0)

    def normalize_frequencies(self):
        self.files["frequencies"] = (self.files["frequencies"] - 1) / 299 * 2 - 1

    def normalize_frequency_response(self):
        self.files["z_vel_mean_sq"] = self.files["z_vel_mean_sq"].sub(self.out_mean[self.files['frequencies']]).div(self.out_std)

    def normalize_field_solution(self, normalize=True, eps=1e-12):
        self.files["z_vel_abs"] = torch.log(torch.square((self.files["z_vel_abs"])) + eps)
        if normalize is True:
            self.files["z_vel_abs"] = self.files["z_vel_abs"].sub(self.field_mean[self.files['frequencies']].unsqueeze(-1).unsqueeze(-1)).div(
                self.field_std)

    def __len__(self):
        return len(self.files["z_vel_mean_sq"])

    def __getitem__(self, index, flip_x=None, flip_y=None):
        data = {key: self.files[key][index] for key in self.keys}
        if self.freq_sampling is True:
            freq_sampling_idx = torch.randint(self.freq_sampling_limit, (self.config.n_random, ))
            data['z_vel_mean_sq'] = data['z_vel_mean_sq'][freq_sampling_idx]
            if 'z_vel_abs' in self.keys:
                data['z_vel_abs'] = data['z_vel_abs'][freq_sampling_idx]
            data['frequencies'] = data['frequencies'][freq_sampling_idx]
        if self.data_augmentation:
            if 'z_vel_abs' in self.keys:
                data['bead_patterns'], data['z_vel_abs'], data['phy_para'] = flip_condition_img(data['bead_patterns'], data['z_vel_abs'], data['phy_para'], flip_x, flip_y)
            else:
                data['bead_patterns'], _, data['phy_para'] = flip_condition_img(data['bead_patterns'], data['bead_patterns'], data['phy_para'], flip_x, flip_y)

        return data


def save_indices_to_csv(indices, file_path):
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in indices:
            writer.writerow(row)


def flip_condition_normalized(condition, flip_x=True, flip_y=True):
    assert condition.dim() == 1
    condition = condition.clone()
    if flip_x:
        condition[2] = -condition[2]
    if flip_y: 
        condition[1] = -condition[1]
    # print(condition.shape)
    return condition


def flip_img(img, flip_x=True, flip_y=True):
    assert img.dim() == 3
    img = img.clone()
    if flip_x:
        img = img.flip(1)
    if flip_y:
        img = img.flip(2)
    return img


def flip_condition_img(img, img_solution, condition, flip_x=None, flip_y=None, flip_p=0.5):
    if flip_x is None:
        flip_x = torch.rand(1) < flip_p
    if flip_y is None:
        flip_y = torch.rand(1) < flip_p
    # print(f'flip_x: {flip_x}, flip_y: {flip_y}')
    img = flip_img(img, flip_x, flip_y)
    img_solution = flip_img(img_solution, flip_x, flip_y)
    condition = flip_condition_normalized(condition, flip_x, flip_y)
    return img, img_solution, condition


def get_dataloader(args, config, logger, num_workers=1, shuffle=True, normalization=True, load_idx=None):
    batch_size = args.batch_size
    np.random.seed(args.seed), torch.cuda.manual_seed_all(args.seed), torch.manual_seed(args.seed)
    generator = torch.Generator(device='cpu')
    if load_idx is None:
        idx = torch.randperm(config.n_samples).numpy()
        freq_idx = torch.stack([torch.randperm(config.n_freqs)[:config.freq_limit] for _ in range(config.n_train_samples)]).numpy()
    else:
        idx, freq_idx = load_idx
    
    train_idx = idx[:config.n_train_samples]
    if config.random_split is False: # this means we should have separate train and val datasets
        assert config.data_path_val != config.data_path_train, "data_path_val should be different from data_path_train"
        trainset = HDF5Dataset(config, config.data_path_train, normalization=normalization, test=False, \
                            sample_idx=train_idx, freq_idx=freq_idx)
        valset = HDF5Dataset(config, config.data_path_val, normalization=normalization, \
                            sample_idx=torch.arange(500)[:config.n_val_samples], test=True) 
    elif config.random_split is True:
        # we assume that the validation set is a subset of the training set
        assert config.train_samples + config.val_samples  <= config.n_samples, "train_samples + val_samples should be less than or equal to n_samples"
        trainset = HDF5Dataset(config, config.data_path_train, normalization=normalization, test=False, \
                            sample_idx=train_idx, freq_idx=freq_idx)
        valset = HDF5Dataset(config, config.data_path_val, normalization=normalization, \
                            sample_idx=idx[-config.n_val_samples:], test=True)
    if config.data_paths_test is not None:
        testset = HDF5Dataset(config, config.data_paths_test, normalization=normalization, test=True)
    else:
        print_log("use valset for testing", logger=logger)
        testset = valset
    if logger is not None:
        idx
        save_indices_to_csv(idx.reshape(len(idx), 1), os.path.join(args.dir, 'sample_indices.csv'))
        save_indices_to_csv(freq_idx, os.path.join(args.dir, 'freq_indices.csv'))

    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size=batch_size,
                                              drop_last=shuffle,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              pin_memory=True,
                                              persistent_workers=True,
                                              generator=generator)
    valloader = torch.utils.data.DataLoader(valset, batch_size=2, drop_last=False, shuffle=False, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=2, drop_last=False, shuffle=False, num_workers=num_workers)
    return trainloader, valloader, testloader, trainset, valset, testset


def extract_mean_std(dataset):

    def get_base_dataset(dataset):
        while hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        return dataset

    base_dataset = get_base_dataset(dataset)
    out_mean, out_std = base_dataset.out_mean, base_dataset.out_std
    field_mean, field_std = base_dataset.field_mean, base_dataset.field_std
    return out_mean, out_std, field_mean, field_std
