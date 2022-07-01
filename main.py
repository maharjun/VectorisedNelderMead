import set_python_path
from typing import Callable, Iterable
import numpy as np

from matplotlib import pyplot as plt

from vectorized_nelder_mead import NelderMeadOptimizer, NelderMeadResult

class rastrigin:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.bounds = [-5.2, 5.2]

    def __call__(self, X):
        return 10*self.input_dim + np.sum(X**2 - 10*np.cos(2*np.pi*X), axis=-1)

def visualize_results(optimization_results: NelderMeadResult, subset_inds, func):

    intermediate_losses = optimization_results.intermediate.losses[subset_inds]
    intermediate_coords = optimization_results.intermediate.coords[subset_inds]

    plt.plot(intermediate_losses.T)
    plt.savefig('loss_convergence_plot.p')
    plt.close()

    def func_wrapper(x):
        """wrapper for arbitrary dimensional x"""
        x_flat = np.reshape(x, (-1, x.shape[-1]))
        func_flat = func(x_flat)
        return np.reshape(func_flat, x.shape[:-1])

    n_steps = intermediate_losses.shape[1] - 1
    x_bounds = func.bounds
    resolution = 200
    plotstep = (x_bounds[1] - x_bounds[0])/resolution

    for i in range(n_steps + 1):
        X, Y = np.meshgrid(np.linspace(x_bounds[0], x_bounds[1], resolution),
                           np.linspace(x_bounds[0], x_bounds[1], resolution), indexing='xy')
        XY = np.stack([X, Y], axis=-1)
        plt.imshow(func_wrapper(XY), extent=[x_bounds[0]-plotstep/2, x_bounds[1]+plotstep/2,
                                             x_bounds[0]-plotstep/2, x_bounds[1]+plotstep/2],
                   origin='lower', cmap='magma_r')
        plt.scatter(intermediate_coords[:, i, 0],
                    intermediate_coords[:, i, 1])
        plt.savefig(f'scatter_for_iter_{i}.svg')
        plt.close()

def get_random_initial_points(size: Iterable[int], rng: np.random.RandomState,
                              lower_bounds: Iterable[np.ndarray],
                              upper_bounds: Iterable[np.ndarray]) -> np.ndarray:
    assert len(lower_bounds) == len(upper_bounds), "The lower_bounds and upper_bounds should be of the same shape"
    assert all(x <= y for x, y in zip(lower_bounds, upper_bounds)), "The lower_bounds should be <= upper_bounds"

    output_dim = len(lower_bounds)
    size = tuple(size)

    rand_points = rng.uniform(size=size + (output_dim,), low=np.array(lower_bounds), high=np.array(upper_bounds))

    return rand_points


if __name__ == '__main__':
    SEED: int = ...
    INPUT_DIM: int = 2
    MAX_STEPS: int = 50
    M: int = 80

    func = rastrigin(INPUT_DIM)
    nelder_mead_params: dict = {
        'return_intermediates': True,
    }
    random_init_points_params: dict = ...

    NelderMeadOptimizer(INPUT_DIM, MAX_STEPS, initial_jitter=None, return_intermediates=True, ftol=1e-2, perform_shrink=True)
    optimizer = NelderMeadOptimizer(**nelder_mead_params)

    grng = np.random.RandomState(seed=SEED)
    initial_points = get_random_initial_points(M, grng,
                                               lower_bounds=[func.bounds[0]]*INPUT_DIM,
                                               upper_bounds=[func.bounds[1]]*INPUT_DIM)

    minimize_result: NelderMeadResult = optimizer.minimize(func, initial_points)

    visualize_results(minimize_result)