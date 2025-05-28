import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np
from scipy import interpolate
from plate_optim.metrics.manufacturing import (
    check_beading_size,
    check_beading_space,
    check_beading_height,
    check_boundary_condition,
    check_derivative,
    DEFAULT_MIN_LENGTH_SCALE,
)


def contour_plot(grid, values, ax=None, cbar=False, cmap="viridis", **kwargs):
    xx = grid.meshgrid_mat[0]
    yy = grid.meshgrid_mat[1]

    if values.ndim == 2:
        zz = values
    else:
        zz = grid.reshape_data(values)

    if ax is None:
        fig, ax = plt.subplots()

    ax.grid(False)
    cs = ax.contourf(xx, yy, zz, levels=50, cmap=cmap, **kwargs)

    if cbar:
        cbar = plt.colorbar(cs, ax=ax)

    return ax


def get_all_comb(*vector):

    n_factorial = 1
    for sample in vector:
        n_factorial = n_factorial * len(sample)

    nodes = np.zeros((n_factorial, len(vector)))

    meshgrid_mat = np.meshgrid(*vector)

    for idx, parameter in enumerate(meshgrid_mat):
        nodes[:, idx] = parameter.flatten()

    return nodes


def calc_fig_size(nrows=1, ncols=1, width_pt=450):
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure heigh
    # https://disq.us/p/2940ij
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inche
    fig_width_in = width_pt * inches_per_pt

    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio
    return (fig_width_in * ncols, fig_height_in * nrows)


def get_multcolumn_subplot(n_regular_plots, n_multi_colm, regular_plot_grid=None, fig_size=None):

    if regular_plot_grid is None:
        if n_regular_plots <= 2:
            n_row = 1
            n_col = 2
            idx_comb = get_all_comb(np.arange(0, 2, dtype=int), np.arange(0, 1, dtype=int))
        elif n_regular_plots <= 4:
            n_row = 2
            n_col = 2
            idx_comb = get_all_comb(np.arange(0, 2, dtype=int), np.arange(0, 2, dtype=int))
        elif n_regular_plots <= 6:
            n_row = 2
            n_col = 3
            idx_comb = get_all_comb(np.arange(0, 3, dtype=int), np.arange(0, 2, dtype=int))
        elif n_regular_plots <= 9:
            n_row = 3
            n_col = 3
            idx_comb = get_all_comb(np.arange(0, 3, dtype=int), np.arange(0, 3, dtype=int))
        elif n_regular_plots <= 16:
            n_row = 4
            n_col = 4
            idx_comb = get_all_comb(np.arange(0, 4, dtype=int), np.arange(0, 4, dtype=int))
    elif isinstance(regular_plot_grid, list):
        n_row = regular_plot_grid[0]
        n_col = regular_plot_grid[1]
        idx_comb = get_all_comb(np.arange(0, n_col, dtype=int), np.arange(0, n_row, dtype=int))

    n_row_total = n_row + n_multi_colm
    if fig_size is None:
        fig_size = calc_fig_size(n_row_total, n_col)
    fig = plt.figure(figsize=fig_size)
    gs = fig.add_gridspec(n_row_total, n_col)

    ax = np.array([])

    for i in range(n_regular_plots):
        ax_elem = fig.add_subplot(gs[int(idx_comb[i, 1]), int(idx_comb[i, 0])])
        ax_elem.grid(True)
        ax = np.append(ax, ax_elem)

    idx_row = n_row
    for i in range(n_multi_colm):
        ax_elem = fig.add_subplot(gs[idx_row, :])
        ax_elem.grid(True)
        ax = np.append(ax, ax_elem)
        idx_row = idx_row + 1

    return fig, ax


class Grid:
    def __init__(self, *dimension_samples, labels=["x", "y", "z"]):

        self.dimension_samples = dimension_samples
        self.labels = labels

        self.n_dim = len(dimension_samples)
        self.n_nodes_per_dim = []
        self.min_per_dim = np.array([])
        self.max_per_dim = np.array([])
        self.l_per_dim = np.array([])
        self.dx_per_dim = np.array([])

        for sample in dimension_samples:
            self.n_nodes_per_dim.append(len(sample))
            self.min_per_dim = np.append(self.min_per_dim, np.min(sample))
            self.max_per_dim = np.append(self.max_per_dim, np.max(sample))
            self.l_per_dim = np.append(self.l_per_dim, np.max(sample) - np.min(sample))
            if sample.size > 1:
                self.dx_per_dim = np.append(self.dx_per_dim, sample[1] - sample[0])

        self.nodes, self.meshgrid_mat = self.full_factorial_grid()
        self.size = self.nodes.shape[0]

    @classmethod
    def create_2D_grid_from_coord(cls, x, y):
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        flag_x_nodes_greater_0 = x > 1e-15
        flag_y_nodes_greater_0 = y > 1e-15
        delta_x = np.min(x[flag_x_nodes_greater_0])
        delta_y = np.min(y[flag_y_nodes_greater_0])

        n_elem_x = int(x_max / delta_x)
        n_elem_y = int(y_max / delta_y)

        x_discretization = np.linspace(x_min, x_max, n_elem_x + 1)
        y_discretization = np.linspace(y_min, y_max, n_elem_y + 1)
        grid = cls(x_discretization, y_discretization)

        return grid

    @classmethod
    def create_3D_grid_from_coord(cls, x, y, z):
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        z_min = np.min(z)
        z_max = np.max(z)

        flag_x_nodes_greater_0 = x > 1e-15
        flag_y_nodes_greater_0 = y > 1e-15
        flag_z_nodes_greater_0 = z > 1e-15
        delta_x = np.min(x[flag_x_nodes_greater_0])
        delta_y = np.min(y[flag_y_nodes_greater_0])
        delta_z = np.min(z[flag_z_nodes_greater_0])

        n_elem_x = int(x_max / delta_x)
        n_elem_y = int(y_max / delta_y)
        n_elem_z = int(z_max / delta_z)

        x_discretization = np.linspace(x_min, x_max, n_elem_x + 1)
        y_discretization = np.linspace(y_min, y_max, n_elem_y + 1)
        z_discretization = np.linspace(z_min, z_max, n_elem_z + 1)
        grid = cls(x_discretization, y_discretization, z_discretization)

        return grid

    def full_factorial_grid(self):
        n_factorial = 1
        for sample in self.dimension_samples:
            n_factorial = n_factorial * len(sample)

        nodes = np.zeros((n_factorial, self.n_dim))
        meshgrid_mat = np.meshgrid(*self.dimension_samples)

        for idx, parameter in enumerate(meshgrid_mat):
            nodes[:, idx] = parameter.flatten()

        return nodes, meshgrid_mat

    def get_oat_sample_mat(self):
        mean_val = []
        for dim_sample in self.dimension_samples:
            mean_val.append(np.mean(dim_sample))

        mean_val = np.array(mean_val)

        sample_mat = []

        for idx, dim_sample in enumerate(self.dimension_samples):
            mean_mat = np.tile(mean_val, (len(dim_sample), 1))
            mean_mat[:, idx] = dim_sample
            sample_mat.append(mean_mat)

        sample_mat = np.vstack(sample_mat)

        return sample_mat

    def reshape_data(self, fun_evals):
        fun_evals_grid = []

        if self.n_dim == 2:
            fun_evals_grid = fun_evals.reshape(self.n_nodes_per_dim[1], self.n_nodes_per_dim[0])
        elif self.n_dim == 3:
            fun_evals_grid = fun_evals.reshape(self.n_nodes_per_dim[1], self.n_nodes_per_dim[0], self.n_nodes_per_dim[2])

        return fun_evals_grid

    def create_interpolator(self, fun_evals):
        interpolator = []

        if self.n_dim == 1:
            interpolator = interpolate.interp1d(self.nodes[:, 0], fun_evals, axis=0)
        elif self.n_dim == 2:
            zz = self.reshape_data(fun_evals=fun_evals)
            interpolator = interpolate.RectBivariateSpline(self.dimension_samples[0], self.dimension_samples[1], zz.T)

        return interpolator

    def create_legend(self, dim=0):
        legendText = []

        for i in range(self.n_nodes_per_dim[dim]):
            legendText.append(f'{self.labels[dim]} = {self.dimension_samples[dim][i] :.2}')

        return legendText

    def sample_picker(self, *condition):

        a = []

        return a

    def get_euclid_distance(self, node_ids, l_max):
        distance_mat = np.ones((self.size, len(node_ids))) * 10000

        for idx, node_id in enumerate(node_ids):
            node_x = self.nodes[node_id, 0]
            node_y = self.nodes[node_id, 1]

            nodes_in_square = self.get_nodes_in_square(node_id, l_max)

            x_coords = self.nodes[nodes_in_square, 0]
            y_coords = self.nodes[nodes_in_square, 1]

            delta_x = x_coords - node_x
            delta_y = y_coords - node_y

            distance_mat[nodes_in_square, idx] = np.sqrt(np.square(delta_x) + np.square(delta_y))

        return distance_mat

    def get_nodes_in_square(self, node_id, l):

        n_nodes_x = l / self.dx_per_dim[0]
        n_idx_x = int(np.floor(n_nodes_x / 2))

        n_nodes_y = l / self.dx_per_dim[1]
        n_idx_y = int(np.floor(n_nodes_y / 2))

        node_ids = np.arange(0, self.size, dtype=int)

        node_ids_mat = self.reshape_data(node_ids)

        node_id_in_mat = np.argwhere(node_ids_mat == node_id)
        x_id = node_id_in_mat[0, 0]
        y_id = node_id_in_mat[0, 1]

        node_ids_in_square = node_ids_mat[np.max((x_id - n_idx_x, 0)):x_id + n_idx_x + 1, np.max((y_id - n_idx_y, 0)):y_id + n_idx_y + 1]

        node_ids_in_square = node_ids_in_square.flatten()

        return node_ids_in_square


def plot_contraints_violations(beading_pattern,min_length_scale=DEFAULT_MIN_LENGTH_SCALE,dimensions=(0.6,0.9),alpha=0.9,ax=None,legend=True):
    if ax is None:
        ax = plt.gca()
    
    metrics = {
        "size": check_beading_size(beading_pattern,dimensions,min_length_scale),
        "space": check_beading_space(beading_pattern,dimensions,min_length_scale),
        "boundary": check_boundary_condition(beading_pattern),
        "height": check_beading_height(beading_pattern),
        "derivative": check_derivative(beading_pattern,height=dimensions[0],width=dimensions[1]),
    }

    cmap = plt.get_cmap("tab10")

    result_rgb = np.zeros((*beading_pattern.shape,3))
    maxval = beading_pattern.max() if beading_pattern.max() > 0 else 1
    result_rgb[:,:,0]= beading_pattern/maxval
    result_rgb[:,:,1] = result_rgb[:,:,0]
    result_rgb[:,:,2] = result_rgb[:,:,0]

    result_rgb*=0.5

    legend_patches = []
    for i,key in enumerate(metrics):
        metrics_color= np.array(cmap(i/10))[:3]
        mask = ~metrics[key]
        result_rgb[mask] = result_rgb[mask]*(1-alpha)+alpha*metrics_color[None,None,:]
        if mask.any(): 
            legend_patches.append(
                mpatches.Patch(color=metrics_color,label=f"{key}: {mask.mean()*100:.2f}%")
            ) 



    
    ax.imshow(result_rgb,extent=[0,dimensions[1],0,dimensions[0]])
    #ax.axis("off")
    if legend:
        ax.legend(handles=legend_patches,
                loc='upper center', bbox_to_anchor=(0.5, -.05),
                ncol=1,
                )




