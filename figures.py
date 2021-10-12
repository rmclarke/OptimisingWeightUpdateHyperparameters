"""Figure-plotting functions.
"""

import json
import multiprocessing
import os

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import plotly.graph_objects as go
import sklearn.gaussian_process as sklgp
import torch as to
import tqdm
from matplotlib.widgets import Button, Slider
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import LogFormatterSciNotation

import util
import parallel_exec as pe


def schedule_evolution(hyperparameter,
                       path,
                       log_scale=False,
                       aggregator=lambda x: x):
    """Using the data in `path`, plot how the schedule of `hyperparameter`
    varies during the pass.
    """
    imported_tags = util.get_tags(path, hyperparameter, '_name')
    data = aggregator(imported_tags[hyperparameter])
    names = imported_tags['_name']
    with open(os.path.join(path, 'config.json'), 'r') as config_file:
        config = json.load(config_file)

    hyperparameter_steps = config['hyperparameter_steps']
    schedules = []
    for index, run_data in enumerate(data):
        if len(run_data) % config['network_weight_steps'] != 0:
            print("Discarding incomplete run at end of data.")
            completed_steps = int(len(run_data) / config['network_weight_steps'])
            data[index] = run_data[:(config['network_weight_steps'] * completed_steps)]
        schedules.append(data[index].view(-1, config['network_weight_steps']))

    fig, ax = plt.subplots()
    if log_scale:
        ax.set_yscale('log')
    plt.subplots_adjust(bottom=0.25)
    trajectory_lines = [plt.plot(schedule[-1], label=name)[0]
                        for schedule, name in zip(schedules, names)]
    plt.legend()
    slider_axes = plt.axes([0.25, 0.1, 0.65, 0.03],
                           facecolor='lightgoldenrodyellow')
    slider = Slider(slider_axes,
                    'Hyperparameter Step',
                    valmin=0,
                    valmax=hyperparameter_steps - 1,
                    valinit=hyperparameter_steps - 1,
                    valstep=1)

    def update_slider(value):
        for line, data in zip(trajectory_lines, schedules):
            try:
                line.set_ydata(data[int(slider.val)])
                line.set_linestyle('-')
            except IndexError:
                line.set_ydata(data[-1])
                line.set_linestyle(':')
                continue
        fig.canvas.draw_idle()
    slider.on_changed(update_slider)

    plt.show()


def plot_2d_weight_evolution(data):
    """Helper script to take the cached `data` from a two-dimensional run and
    plot a variety of useful graphics.
    """
    true_inv_hessian = to.zeros(2, 2)
    for i in range(2):
        for j in range(2):
            true_inv_hessian[i, j] = data[f'True_Inverse_Hessian/{i},{j}'][0][0]
    true_hessian = true_inv_hessian.inverse()

    num_values = len(data[f'LT_Curvature_Transform/0,0'][0])
    lt_curvature_transform = to.zeros(num_values, 2, 2)
    lt_curvature_transform_grad = to.zeros(num_values, 2, 2)
    for i in range(2):
        for j in range(i+1):
            lt_curvature_transform[:, i, j] = data[f'LT_Curvature_Transform/{i},{j}'][0]
            lt_curvature_transform_grad[(1+data['config'][0]['network_weight_steps']):, i, j] = data[f'LT_Curvature_Transform_Grad/{i},{j}'][0]
    # Catch problematic transforms
    curvature_transform = lt_curvature_transform @ lt_curvature_transform.transpose(1, 2)
    singular_transforms = (curvature_transform.det() == 0)
    filtered_curvature_transform = curvature_transform
    if singular_transforms.any():
        filtered_curvature_transform = curvature_transform.clone()
        filtered_curvature_transform[singular_transforms] = float('nan')
    inv_curvature_transform = curvature_transform.inverse()

    num_values = len(data[f'Weights/0'][0])
    weights = to.zeros(num_values, 2)
    for i in range(2):
        weights[:, i] = data[f'Weights/{i}'][0]

    X, Y = np.meshgrid(
        np.linspace(-5, 5, 100),
        np.linspace(-5, 5, 100))

    Z = (true_hessian[0, 0] * X**2
         + (true_hessian[0, 1] + true_hessian[1, 0])*X*Y
         + true_hessian[1, 1] * Y**2)

    X_hat = X[None, :]
    Y_hat = Y[None, :]
    inv_curvature_transform = inv_curvature_transform.numpy()
    weights = weights.numpy()
    Z_hat = (inv_curvature_transform[:, 0:1, 0:1] * (X_hat - weights[:, 0:1, None])**2
             + (inv_curvature_transform[:, 0:1, 1:2] + inv_curvature_transform[:, 1:2, 0:1]) * (X_hat - weights[:, 0:1, None]) * (Y_hat - weights[:, 1:2, None])
             + inv_curvature_transform[:, 1:2, 1:2] * (Y_hat - weights[:, 1:2, None])**2)

    hessian_weight_product = to.from_numpy(weights) @ true_hessian
    transform_normal = to.stack([hessian_weight_product[:, 1]**2,
                                 -hessian_weight_product.prod(dim=1),
                                 hessian_weight_product[:, 0]**2],
                                dim=1)
    transform_normal = transform_normal / transform_normal.norm(dim=1).unsqueeze(1)
    flat_curvature_transform = to.stack([curvature_transform[:, 0, 0],
                                         curvature_transform[:, 0, 1],
                                         curvature_transform[:, 1, 1]],
                                        dim=1)
    transform_offset = to.stack([weights[:, 0] / hessian_weight_product[:, 0],
                                 to.zeros_like(hessian_weight_product[:, 0]),
                                 weights[:, 1] / hessian_weight_product[:, 1]],
                                dim=1)
    transform_distance = (to.norm(to.cross(transform_offset - flat_curvature_transform,
                                           transform_normal, dim=1), dim=1)
                          / to.norm(transform_normal, dim=1))
    if (weights[::2] == weights[0]).all():
        transform_distance[1::2] = float('nan')

    fig = plt.gcf()
    master_grid = fig.add_gridspec(2, 1,
                                   height_ratios=(20, 1))
    plot_grid = master_grid[0].subgridspec(2, 3)
    controls_grid = master_grid[1].subgridspec(1, 3,
                                               width_ratios=(15, 1, 1))

    data_ax = fig.add_subplot(plot_grid[:, 1])
    data_ax.set_xlim(-5, 5)
    data_ax.set_ylim(-5, 5)
    slider_axes = fig.add_subplot(controls_grid[0, 0],
                                  facecolor='lightgoldenrodyellow')
    minus_button_axes = fig.add_subplot(controls_grid[0, 1])
    plus_button_axes = fig.add_subplot(controls_grid[0, 2])
    slider = Slider(slider_axes,
                    'Update Step',
                    valmin=0,
                    valmax=len(weights) - 1,
                    valinit=0,
                    valstep=1)
    plus_button = Button(plus_button_axes, '+')
    plus_button.on_clicked(lambda *_: slider.set_val(slider.val + 1))
    minus_button = Button(minus_button_axes, '–')
    minus_button.on_clicked(lambda *_: slider.set_val(slider.val - 1))

    transform_ax = fig.add_subplot(plot_grid[0, 0])
    transform_log_ax = transform_ax.twinx()
    # transform_log_ax.set_yscale('log')
    transform_log_ax.plot(transform_distance,
                          color='k',
                          label=r'$d_{2}(\mathbf{T}, \mathbf{H})$',
                          marker='.',
                          markersize=2)
    lt_true_inv_hessian = true_inv_hessian.cholesky()
    for i in range(2):
        for j in range(i + 1):
            learned_trend = transform_ax.plot(lt_curvature_transform[:, i, j],
                                              label=f'[{i}, {j}]')
            transform_ax.axhline(lt_true_inv_hessian[i, j],
                                 linestyle='--',
                                 color=learned_trend[0].get_color())
    current_transform_lines = [transform_ax.axvline(0, color='r')]
    transform_ax.set_xlim(0, len(weights))
    transform_ax.legend()
    transform_ax.set_xlabel('Update Step')
    transform_ax.set_ylabel('Learned Lower-Triangular Curvature_Transform Value')
    transform_log_ax.set_ylabel('Curvature_Transform Distance from Optimal Space')
    transform_ax.set_title('Evolution of Lower-Triangular Curvature_Transform')

    grad_ax = fig.add_subplot(plot_grid[1, 0])
    for i in range(2):
        for j in range(i + 1):
            grad_ax.plot(-lt_curvature_transform_grad[:, i, j])
    current_transform_lines.append(grad_ax.axvline(0, color='r'))
    grad_ax.set_xlim(0, len(weights))
    grad_ax.set_xlabel('Update Step')
    grad_ax.set_ylabel('Lower-Triangular Curvature_Transform Negative Gradient')
    grad_ax.set_title('Evolution of Lower-Triangular Curvature_Transform Negative Gradients')

    curvature_transform_ax = fig.add_subplot(plot_grid[:, 2],
                                             projection='3d')
    current_lt_curvature_transform_point = curvature_transform_ax.plot([0], [0], [0], 'ro')[0]
    optimal_curvature_transform_point = curvature_transform_ax.plot(
        lt_true_inv_hessian[0, 0],
        lt_true_inv_hessian[1, 0],
        lt_true_inv_hessian[1, 1],
        'ko')
    curvature_transform_ax.plot(
        lt_curvature_transform[:, 0, 0],
        lt_curvature_transform[:, 1, 0],
        lt_curvature_transform[:, 1, 1],
        '.-',
        markersize=2)
    optimal_loci = [curvature_transform_ax.plot([0, 0], [0, 0], [0, 0],
                                                '.', markersize=2)[0]
                    for _ in range(data['config'][0]['network_weight_steps'])]
    transform_update_arrow = None

    curvature_transform_ax.set_xlim(-5, 5)
    curvature_transform_ax.set_ylim(-5, 5)
    curvature_transform_ax.set_zlim(-5, 5)

    def update_slider(value):
        nonlocal transform_update_arrow
        last_update = weights[value-1] - -inv_curvature_transform[value-1] @ (weights[value] - weights[value-1])

        old_xlim = data_ax.get_xlim()
        old_ylim = data_ax.get_ylim()
        data_ax.cla()
        contours = data_ax.contour(X, Y, Z)
        data_ax.plot(0, 0, 'ko')
        contours_hat = data_ax.contour(X, Y, Z_hat[value])
        data_ax.clabel(contours, inline=True)
        data_ax.clabel(contours_hat, inline=True)
        data_ax.plot(*weights.T, '.-')
        data_ax.plot(*weights[0], 'o')
        data_ax.plot(*weights[value], 'ro')
        data_ax.annotate("", xy=weights[value], xytext=weights[value-1],
                         arrowprops={'arrowstyle': "->",
                                     'color': 'r'})
        data_ax.annotate("", xy=last_update, xytext=weights[value-1],
                         arrowprops={'arrowstyle': "->",
                                     'color': 'g'})
        data_ax.set_xlim(*old_xlim)
        data_ax.set_ylim(*old_ylim)
        data_ax.set_aspect('equal')
        data_ax.set_xlabel('Weight 0')
        data_ax.set_ylabel('Weight 1')
        data_ax.set_title('Evolution of Weights')

        for line in current_transform_lines:
            line.set_xdata((value, value))

        base_value = value - (value % (data['config'][0]['network_weight_steps'] + 1))
        for line_id, line in enumerate(optimal_loci):
            line_value = base_value + line_id
            line_vars = np.linspace(-10, 10, 10000)
            line_points = transform_offset[line_value] + line_vars[:, None]*transform_normal[line_value].numpy()
            # +ve sqrt for x, +ve sqrt for z
            base_lt_line_x = lt_line_x = np.sqrt(line_points[:, 0])
            base_lt_line_y = lt_line_y = line_points[:, 1] / lt_line_x
            base_lt_line_z = lt_line_z = np.sqrt(line_points[:, 2] - lt_line_y**2)
            # -ve sqrt for x, +ve sqrt for z
            lt_line_x = np.concatenate((lt_line_x, -base_lt_line_x))
            lt_line_y = np.concatenate((lt_line_y, -base_lt_line_y))
            lt_line_z = np.concatenate((lt_line_z, base_lt_line_z))
            # +ve sqrt for x, -ve sqrt for z
            lt_line_x = np.concatenate((lt_line_x, base_lt_line_x))
            lt_line_y = np.concatenate((lt_line_y, base_lt_line_y))
            lt_line_z = np.concatenate((lt_line_z, -base_lt_line_z))
            # -ve sqrt for x, -ve sqrt for z
            lt_line_x = np.concatenate((lt_line_x, -base_lt_line_x))
            lt_line_y = np.concatenate((lt_line_y, -base_lt_line_y))
            lt_line_z = np.concatenate((lt_line_z, -base_lt_line_z))

            line.set_data_3d(lt_line_x, lt_line_y, lt_line_z)
            if line_value == value:
                break

        for dead_line in optimal_loci[line_id+1:]:
            dead_line.set_data_3d([0, 0], [0, 0], [0, 0])
        flat_current_lt_curvature_transform = to.stack(
            [lt_curvature_transform[value][0, 0],
             lt_curvature_transform[value][1, 0],
             lt_curvature_transform[value][1, 1]])
        if transform_update_arrow:
            transform_update_arrow.remove()
            transform_update_arrow = None
        if (value % (data['config'][0]['network_weight_steps'] + 1)) == data['config'][0]['network_weight_steps']:
            # About to do a hyperparameter update, so indicate it
            flat_next_lt_curvature_transform = to.stack(
                [lt_curvature_transform[value+1][0, 0],
                 lt_curvature_transform[value+1][1, 0],
                 lt_curvature_transform[value+1][1, 1]])
            transform_update_arrow = curvature_transform_ax.quiver(
                *flat_current_lt_curvature_transform,
                *(flat_next_lt_curvature_transform - flat_current_lt_curvature_transform),
                color='red',
                # Large scale factor to clearly show direction
                length=10000*(flat_next_lt_curvature_transform - flat_current_lt_curvature_transform).norm().item())

        current_lt_curvature_transform_point.set_data_3d(
            lt_curvature_transform[value][0, 0],
            lt_curvature_transform[value][1, 0],
            lt_curvature_transform[value][1, 1])

        plt.gcf().canvas.draw_idle()
    slider.on_changed(update_slider)
    update_slider(0)

    plt.show()


def plot_2d_eigen_evolution(data):
    true_inv_hessian = to.zeros(2, 2)
    for i in range(2):
        for j in range(2):
            true_inv_hessian[i, j] = data[f'True_Inverse_Hessian/{i},{j}'][0][0]
    true_hessian = true_inv_hessian.inverse()
    true_eig = true_hessian.symeig(eigenvectors=True)

    num_values = len(data['Curvature_Transform/0,0'][0])
    curvature_transform = to.zeros(num_values, 2, 2)
    for i in range(2):
        for j in range(2):
            curvature_transform[:, i, j] = data[f'Curvature_Transform/{i},{j}'][0]
    inv_curvature_transform = curvature_transform.inverse()
    learned_eig = inv_curvature_transform.symeig(eigenvectors=True)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(np.arctan(learned_eig.eigenvectors[:, 1, :]
                      / learned_eig.eigenvectors[:, 0, :]),
            learned_eig.eigenvalues, '.-')
    for true_idx in range(2):
        point = ax.plot(np.arctan(true_eig.eigenvectors[1, true_idx]
                                  / true_eig.eigenvectors[0, true_idx]),
                        true_eig.eigenvalues[true_idx], 'o')
        ax.plot(np.arctan(true_eig.eigenvectors[1, true_idx]
                          / true_eig.eigenvectors[0, true_idx]).tile(2),
                [0, learned_eig.eigenvalues.max()], color=point[0].get_color())

    plt.show()


def plot_2d_full_objective(data):
    true_inv_hessian = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            true_inv_hessian[i, j] = data[f'True_Inverse_Hessian/{i},{j}'][0][0]
    true_hessian = np.linalg.inv(true_inv_hessian)

    weights_0 = np.stack((data['Weights/0'][0][0],
                          data['Weights/1'][0][0]))

    X, Y, Z = np.meshgrid(
        np.linspace(-0.5, 0.5, 100),
        np.linspace(-0.5, 0.5, 100),
        np.linspace(-0.5, 0.5, 100))

    curvature_transforms = np.zeros((*X.shape, 2, 2))
    curvature_transforms[..., 0, 0] = X
    curvature_transforms[..., 1, 1] = Y
    curvature_transforms[..., 0, 1] = curvature_transforms[..., 1, 0] = Z

    weights_1 = weights_0 - curvature_transforms @ true_hessian @ weights_0
    objective = weights_1[..., None, :] @ true_hessian @ weights_1[..., :, None]

    # fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    # ax.scatter(X, Y, Z, c=objective, alpha=0.1)
    # ax.set_xlabel("X")
    # ax.set_ylabel("Y")
    # ax.set_zlabel("Z")
    # plt.show()

    figure = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=objective.flatten(),
        opacity=0.1,
        opacityscale="min",
        isomax=20,
        isomin=0,
        surface_count=25
    ))
    figure.add_trace(
        go.Scatter3d(
            x=[true_inv_hessian[0,0]],
            y=[true_inv_hessian[1,1]],
            z=[true_inv_hessian[0,1]],
        )
    )
    figure.show()


def plot_average_comparison(root_directory, key):
    """Average over the contents of each subdirectory of `root_directory`, plotting
    comparisons.
    """
    category_directories = [entry.path for entry in os.scandir(root_directory)
                            if entry.is_dir()]
    category_directories.sort()
    all_data = []
    for category_directory in category_directories:
        category_data = util.get_tags(category_directory)
        stacked_data = to.stack(category_data[key])
        category_means = stacked_data.mean(dim=0)
        category_stds = stacked_data.std(dim=0)

        all_data.append(stacked_data[:, 100])
        # plt.plot(category_means, label=category_directory)
        # plt.fill_between(range(len(category_means)),
        #                  category_means - category_stds,
        #                  category_means + category_stds,
        #                  alpha = 0.2)

    plt.boxplot(all_data, labels=category_directories)
    # plt.legend()
    plt.yscale('log')
    plt.xticks(rotation=15)
    plt.show()


def compute_condition_numbers(true_hessian, weights):
    """Compute the condition number implied by solving the
    exact-optimum finding problem for the given parameters.
    """
    malformed_coefficients = to.from_numpy(weights) @ true_hessian
    num_hessian_elements = int(weights.shape[1] * (weights.shape[1] + 1) / 2)
    coefficient_matrix = to.zeros(weights.shape[0], weights.shape[1], num_hessian_elements)

    coefficient_id = 0
    for row_id in range(weights.shape[1]):
        for column_id in range(row_id, weights.shape[1]):
            coefficient_matrix[:, row_id, coefficient_id] = malformed_coefficients[:, column_id]
            coefficient_matrix[:, column_id, coefficient_id] = malformed_coefficients[:, row_id]
            coefficient_id += 1

    full_coefficients = to.cat(list(coefficient_matrix), dim=0).numpy()
    result = [np.linalg.cond(full_coefficients[:2*i])
              for i in range(2, len(weights)+1)]
    return result
    # return np.linalg.cond(full_coefficients.numpy())


def parse_data_by_algorithm(data,
                            metric,
                            delete_nans=True,
                            nearly_nan_threshold=None,
                            add_random_batch=True,
                            performance_function=to.min,
                            exclude_algorithms=[],
                            include_times=False):
    """Extract and parse the `metric` data from `data`. `exclude_algorithms`
    must only be used to avoid reading algorithms which don't exist; NOT to
    selectively not read algorithm data we don't care about.
    """
    algorithms = [config['algorithm'] for config in data['config']
                  if config['algorithm'] not in exclude_algorithms]
    metric_data = {}
    metric_time_data = {}
    sorted_algorithms = sorted(
        list(set(algorithms)),
        key=lambda alg: pe.ALL_ALGORITHMS.index(alg) if alg in pe.ALL_ALGORITHMS else -1)
    if add_random_batch:
        sorted_algorithms.insert(sorted_algorithms.index('Random_SteppedLR') + 1,
                                 'Random_3Batched')
        random_final_validation_losses = to.stack([
            trajectory[-1] for this_algorithm, trajectory in zip(algorithms, data['Loss/Validation'])
            if this_algorithm == 'Random_Validation'])
    for algorithm in sorted_algorithms:
        if algorithm == 'Random_3Batched':
            random_data = metric_data['Random_Validation']
            indices = np.random.permutation(len(random_data))
            metric_data[algorithm] = to.stack([
                random_data[[batch_indices]][
                    performance_function(
                        random_final_validation_losses[[batch_indices]], dim=0)
                    .indices]
                for batch_indices in zip(indices[0::3], indices[1::3], indices[2::3])
            ])
            if include_times:
                metric_time_data[algorithm] = to.stack([
                    metric_time_data['Random_Validation'][[batch_indices]][
                        performance_function(
                            random_final_validation_losses[[batch_indices]], dim=0)
                        .indices]
                    for batch_indices in zip(indices[0::3], indices[1::3], indices[2::3])
                ])
            metric_data.pop('Random_Validation')
            continue
        algorithm_data = [
            trajectory for this_algorithm, trajectory in zip(algorithms, data[metric])
            if this_algorithm == algorithm]
        if not isinstance(algorithm_data[-1], to.Tensor):
            assert not include_times
            if isinstance(algorithm_data[-1], (float, int)):
                metric_data[algorithm] = to.tensor(algorithm_data)
            else:
                metric_data[algorithm] = algorithm_data
        else:
            if include_times:
                time_data = [
                    times for this_algorithm, times in zip(algorithms, data[metric + '/wall_time'])
                    if this_algorithm == algorithm]
            max_shape = max([run.shape for run in algorithm_data])
            for run_id, run_data in enumerate(algorithm_data):
                # Deal with any truncated series
                if run_data.shape != max_shape:
                    extended_data = to.full(max_shape, float('nan'))
                    extended_data[:len(run_data)] = run_data
                    algorithm_data[run_id] = extended_data
                    if include_times:
                        run_time_data = time_data[run_id]
                        extended_times = to.full(max_shape, run_time_data.max())
                        extended_times[:len(run_time_data)] = run_time_data
                        time_data[run_id] = extended_times
            metric_data[algorithm] = to.stack(algorithm_data)
            if include_times:
                metric_time_data[algorithm] = to.stack(time_data)
            if delete_nans:
                nan_mask = metric_data[algorithm][:, -1].isnan()
                if nearly_nan_threshold:
                    nan_mask[metric_data[algorithm][:, -1] >= nearly_nan_threshold] = 1
                metric_data[algorithm] = metric_data[algorithm][~nan_mask]
                if include_times:
                    metric_time_data[algorithm] = metric_time_data[algorithm][~nan_mask]
    if include_times:
        return metric_data, metric_time_data
    else:
        return metric_data


def _median_aggregator(indices, algorithm_data):
    return np.nanmedian(algorithm_data[indices], axis=0)


def plot_evolution_envelope(data,
                            metric,
                            delete_nans=True,
                            nearly_nan_threshold=None,
                            legend=True,
                            wall_time=False,
                            ignore_algorithms=[]):
    """Plot the average trends of `metric`, enveloped by their min/max.
    """
    bootstrapped_data = {}
    random_present = any(config['algorithm'] == 'Random_Validation'
                         for config in data['config'])
    metric_data = parse_data_by_algorithm(data,
                                          metric,
                                          delete_nans=delete_nans,
                                          nearly_nan_threshold=nearly_nan_threshold,
                                          include_times=wall_time,
                                          add_random_batch=(not wall_time
                                                            and random_present))
    if wall_time:
        metric_data, time_data = metric_data
        for algorithm in ignore_algorithms:
            time_data.pop(algorithm, None)
    for algorithm in ignore_algorithms:
        metric_data.pop(algorithm, None)
    for algorithm, algorithm_data in tqdm.tqdm(metric_data.items()):
        algorithm_data = algorithm_data.numpy()

        if wall_time:
            algorithm_data, bootstrap_timestamps = util.interpolate_timestamps(
                algorithm_data, time_data[algorithm], num_timestamps=1000)
            data_x = bootstrap_timestamps
        else:
            data_x = range(algorithm_data.shape[1])
        bootstrap_indices = util.bootstrap_sample(len(algorithm_data), 1000)
        bootstrap_samples = algorithm_data[bootstrap_indices]
        bootstrapped_data[algorithm] = bootstrap_samples
        bootstrap_medians = np.nanmedian(bootstrap_samples, axis=0)
        bootstrap_median_mean = np.mean(bootstrap_medians, axis=0)
        bootstrap_median_std = np.nanstd(bootstrap_medians, axis=0)

        plt.plot(data_x, bootstrap_median_mean, label=algorithm)
        plt.fill_between(data_x,
                         bootstrap_median_mean-bootstrap_median_std,
                         bootstrap_median_mean+bootstrap_median_std,
                         alpha=0.4)

    plt.yscale('log')
    plt.xlabel("Runtime (s)" if wall_time else "Update Step")
    plt.ylabel(metric)
    if legend:
        plt.legend()
    plt.show()
    # TODO: Rework logic to return indices instead
    return bootstrapped_data


def plot_toy_ablation_heatmap(root_directory,
                              normaliser=None,
                              with_stds=False,
                              data_extractor=lambda data: to.stack(data['Unnormalised_Loss/Test'])[:, -1].numpy(),
                              aggregator=np.nanmedian,
                              num_format='{:.3f}',
                              value_scale=1):
    heatmap_values = np.full((10, 10), float('nan'))
    heatmap_text = np.full((10, 10), None)
    for segment_directory in os.scandir(root_directory):
        update_interval, rollback_distance = segment_directory.name.split('_')
        update_interval = int(update_interval[6:])
        rollback_distance = int(rollback_distance[8:])

        data_dict = util.get_tags(segment_directory)
        segment_data = data_extractor(data_dict) * value_scale
        aggregated_loss = aggregator(segment_data)
        std_loss = np.nanstd(segment_data)
        heatmap_values[update_interval-1, rollback_distance-1] = aggregated_loss
        label = num_format.format(aggregated_loss)
        if with_stds:
            label += '\n' + num_format.format(std_loss)
        heatmap_text[update_interval-1, rollback_distance-1] = plt.text(
            rollback_distance-1, update_interval-1, label,
            ha='center', va='center', fontsize=8)

    if normaliser is None:
        normaliser = Normalize(vmin=0, vmax=np.nanmax(heatmap_values))
    for value, text in zip(heatmap_values.flat, heatmap_text.flat):
        if normaliser(value) < 0.5:
            text.set_color('w')
    plt.imshow(heatmap_values, origin='lower', norm=normaliser)

    plt.xlabel("Look-Back Distance $i$")
    plt.ylabel("Update Interval $T$")
    plt.xticks(range(10), range(1, 10+1))
    plt.yticks(range(10), range(1, 10+1))
    # plt.title("Median of Runs (Standard Deviation)")
    plt.grid(False)
    plt.gcf().set_size_inches(4.5, 4.5)
    plt.tight_layout()
    plt.show()


def plot_final_cdfs(data,
                    metric,
                    nearly_nan_threshold=None,
                    add_random_batch=True):
    """Plot a CDF of the final value of `metric` reached by all runs in `data`.
    """
    metric_data = parse_data_by_algorithm(data,
                                          metric,
                                          delete_nans=False,
                                          add_random_batch=add_random_batch)
    for algorithm, algorithm_data in metric_data.items():
        if nearly_nan_threshold:
            algorithm_data[algorithm_data[:, -1] >= nearly_nan_threshold] = float('nan')
        algorithm_data = np.sort(algorithm_data[:, -1])
        cdf_values = np.linspace(0, 1, len(algorithm_data)+1)
        # Don't need to double up on first and last repeated values
        plt.plot(np.repeat(algorithm_data, 2),
                 np.repeat(cdf_values, 2)[1:-1],
                 label=algorithm)
    plt.show()


def plot_runtime_violins(data, exclude_algorithms=[]):
    """Plot violin plots of the runtime of all runs in `data`."""
    all_start_times = parse_data_by_algorithm(data,
                                              'first_timestamp',
                                              add_random_batch=False)
    all_stop_times = parse_data_by_algorithm(data,
                                             'last_timestamp',
                                             add_random_batch=False)
    all_losses = parse_data_by_algorithm(data,
                                         'Loss/Test',
                                         delete_nans=False,
                                         add_random_batch=False)
    for algorithm_id, (start_times, stop_times, losses) in enumerate(
            zip(all_start_times.values(), all_stop_times.values(), all_losses.values())):
        if tuple(all_losses.keys())[algorithm_id] in exclude_algorithms:
            continue
        valid_runs = ~to.isnan(losses[:, -1])
        runtimes = stop_times[valid_runs] - start_times[valid_runs]
        violin = plt.violinplot([runtimes], positions=[algorithm_id])
        colour = violin['cbars'].get_color()
        plt.scatter(algorithm_id, np.mean(runtimes.numpy()), marker='o', c=colour)
        plt.scatter(algorithm_id, np.median(runtimes.numpy()), marker='x', c=colour)


def plot_hyperparameter_evolution(data,
                                  heatmap_algorithm,
                                  trajectory_algorithm,
                                  hyperparameter_x,
                                  hyperparameter_y,
                                  metric,
                                  num_trajectories,
                                  hyperparameter_x_exclude_algorithms=[],
                                  hyperparameter_y_exclude_algorithms=[],
                                  hyperparameter_x_init_transform=lambda x: x,
                                  hyperparameter_y_init_transform=lambda y: y,
                                  metric_transform=lambda z: z):
    """Based on the runs in `data`, construct a heatmap of final performance on
    `metric` from `heatmap_algorithm`, then superimpose `hyperparameters`
    evolution from `trajectory_algorithm`.
    """
    config_data = parse_data_by_algorithm(
        data, 'config', add_random_batch=False)
    metric_data = parse_data_by_algorithm(
        data, metric, add_random_batch=False, delete_nans=False)
    hyperparameters_x = parse_data_by_algorithm(
        data,
        f'Hyperparameter/{hyperparameter_x.title()}',
        exclude_algorithms=hyperparameter_x_exclude_algorithms,
        delete_nans=False,
        add_random_batch=False)
    hyperparameters_y = parse_data_by_algorithm(
        data,
        f'Hyperparameter/{hyperparameter_y.title()}',
        exclude_algorithms=hyperparameter_y_exclude_algorithms,
        delete_nans=False,
        add_random_batch=False)

    # Background heatmap
    heatmap_points = [[], []]
    heatmap_values = []
    for run_config, run_metric in zip(config_data[heatmap_algorithm],
                                      metric_data[heatmap_algorithm]):
        run_init = run_config['config_dicts']['network_weight']
        heatmap_points[0].append(run_init[hyperparameter_x])
        heatmap_points[1].append(run_init[hyperparameter_y])
        heatmap_values.append(run_metric[-1])

    heatmap_points = np.array(heatmap_points).T
    heatmap_values = np.array(heatmap_values)
    nan_replacement = 150  # BayesOpt NaN max replacement
    heatmap_values[np.isnan(heatmap_values)] = nan_replacement
    heatmap_size = (100, 100)
    heatmap_extents = np.array([[-6, -0.5],
                                [-7, 1]])
    kernel = (sklgp.kernels.RBF()
              + sklgp.kernels.WhiteKernel())
    gaussian_process = sklgp.GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True
    )
    gaussian_process.fit(heatmap_points, np.log10(heatmap_values))
    x_range = np.logspace(*heatmap_extents[0], heatmap_size[0])
    y_range = np.logspace(*heatmap_extents[1], heatmap_size[1])
    heatmap_points = np.column_stack(
        [dim.ravel() for dim in np.meshgrid(x_range, y_range)])
    heatmap_predictions = 10**gaussian_process.predict(np.log10(heatmap_points))
    heatmap_predictions = heatmap_predictions.clip(max=nan_replacement)
    heatmap = plt.pcolormesh(*np.meshgrid(x_range, y_range),
                             heatmap_predictions.reshape(*heatmap_size),
                             shading='nearest',
                             rasterized=True,
                             norm=LogNorm())
    colourbar = plt.colorbar(heatmap)

    class NaNLogFormatter(LogFormatterSciNotation):
        def __call__(self, x, pos=None):
            if x == nan_replacement:
                return 'NaN'
            else:
                return super().__call__(x, pos)

    colourbar.set_ticks(colourbar.get_ticks().tolist() + [nan_replacement])
    colourbar.ax.yaxis.set_major_formatter(NaNLogFormatter())

    # Hyperparameter trajectories
    valid_mask = (np.isfinite(hyperparameters_x[trajectory_algorithm][:, -1])
                  * np.isfinite(hyperparameters_y[trajectory_algorithm][:, -1]))
    hyperparameters_x[trajectory_algorithm] = hyperparameters_x[trajectory_algorithm][valid_mask]
    hyperparameters_y[trajectory_algorithm] = hyperparameters_y[trajectory_algorithm][valid_mask]
    # These logs are already transformed, so no need to re-transform
    trajectory_indices = np.random.choice(
        len(hyperparameters_x[trajectory_algorithm]),
        size=num_trajectories,
        replace=False)
    trajectory_lines = plt.plot(
        hyperparameters_x[trajectory_algorithm][trajectory_indices].T,
        hyperparameters_y[trajectory_algorithm][trajectory_indices].T)
    # Start points
    plt.scatter(hyperparameters_x[trajectory_algorithm][trajectory_indices][:, 0],
                hyperparameters_y[trajectory_algorithm][trajectory_indices][:, 0],
                c=[t.get_color() for t in trajectory_lines],
                marker='o')
    # End points
    plt.scatter(hyperparameters_x[trajectory_algorithm][trajectory_indices][:, -1],
                hyperparameters_y[trajectory_algorithm][trajectory_indices][:, -1],
                c=[t.get_color() for t in trajectory_lines],
                marker='s')
    add_arrows(*trajectory_lines)

    plt.show()


def add_arrows(*lines, num_arrows=4, size=5, index_spread=10):
    """Add `num_arrows` equally-spaced arrows of `size` to each of `lines`,
    based on their existing paths.
    Heavily inspired by https://stackoverflow.com/a/34018322
    """
    for line in lines:
        colour = line.get_color()
        points = line.get_xydata()
        points = points[~np.isnan(points).any(axis=1)]

        # Transform points into log space if necessary for accurate lengths
        len_points = points.copy()
        if plt.gca().get_xscale() == 'log':
            len_points[:, 0] = np.log10(len_points[:, 0])
        if plt.gca().get_yscale() == 'log':
            len_points[:, 1] = np.log10(len_points[:, 1])

        cumulative_length = np.linalg.norm(
            len_points[1:] - len_points[:-1],
            axis=1).cumsum()
        # Exclude start and end of line
        arrow_positions = np.linspace(0, cumulative_length[-1],
                                      num=(num_arrows+2))[1:-1]
        start_indices = np.sum(
            cumulative_length[:, None] < arrow_positions[None, :],
            axis=0)

        for start_index in start_indices:
            start_index = max(start_index - index_spread, 0)
            end_index = min(start_index + index_spread, len(points) - 1)
            line.axes.annotate('',
                               xytext=points[start_index],
                               xy=points[end_index],
                               arrowprops={'arrowstyle': 'simple',
                                           'color': colour},
                               size=size)
