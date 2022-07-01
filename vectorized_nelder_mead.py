from typing import NamedTuple, Optional
import numpy as np
from collections import namedtuple


class NelderMeadIntermediates(NamedTuple):
    """The named tuple that containts the intermediates from the nelder-mead
    optimization

    Attributes
    ==========

    points: np.ndarray of size (M, max_steps+1, dim)
        points[i, :, :] contains the coordinate of the best loss in the simplex for
        each step of optimization (including the initial simplex) starting from the
        i'th initial point. Note that if there are fewer than max_steps of
        optimization, all further values are equal to the minima

    losses: np.ndarray of size (M, max_steps+1)
        losses[i, :] contains the best value in the simplex for each step of
        optimization (including the initial simplex) starting from the i'th initial
        point. Note that if there are fewer than max_steps of optimization, all
        further values are equal to the minima

    n_steps: integer np.ndarray of size (M,)
        n_steps[i] is the number of optimization steps performed starting from the i'th
        initial point"""

    coords: np.ndarray
    losses: np.ndarray
    n_steps: np.ndarray

class NelderMeadResult(NamedTuple):
    """The named tuple that contains the result of the minimization

    Attributes
    ==========

    x: np.ndarray
        An (M, dim) array containing the final optimized positions corresponding to
        each of the M initial points

    loss: np.ndarray
        An (M,) array containing the final optimized loss value

    intermediate: NelderMeadIntermediates
        An instance of NelderMeadIntermediates that contains the intermediate points in the optimization
    """
    x: np.ndarray
    loss: np.ndarray
    intermediate: NelderMeadIntermediates


class _NMAlgorithmState(NamedTuple):
    """
    Contains all information that is calculated and required each iteration of the nelder-mead loop
    """
    simplex: np.ndarray
    loss: np.ndarray

    coord_best: np.ndarray = None
    coord_worst: np.ndarray = None
    coord_secondworst: np.ndarray = None
    coord_centroid: np.ndarray = None

    loss_best: np.ndarray = None
    loss_worst: np.ndarray = None
    loss_secondworst: np.ndarray = None

    coord_reflected: np.ndarray = None
    loss_reflected: np.ndarray = None


class NelderMeadOptimizer:

    def __init__(self, input_dim, max_steps, initial_jitter=None, return_intermediates=False, perform_shrink=True):
        self.input_dim = input_dim
        self.max_steps = max_steps
        self.initial_jitter = initial_jitter
        self.return_intermediates = return_intermediates

        if initial_jitter is not None:
            initial_jitter = np.asarray(initial_jitter, dtype=np.float32)
            assert initial_jitter.ndim == 0 or initial_jitter.shape == (self.input_dim,), \
                (f"initial_jitter shape {self.initial_jitter} incompatible"
                 f" with specified input_dim {self.input_dim}")
            assert np.all(initial_jitter > 0), \
                "initial_jitter must be None or a float array-like with positive values"
        else:
            self.initial_jitter = None # relative jitter with 5% jitter


    def _get_initial_state(self, func, initial_points):
        dim = self.input_dim
        M = len(initial_points)

        if self.initial_jitter is None:
            initial_simplex_jitter = initial_points * 0.05  # (M, dim)
            assert initial_simplex_jitter.shape == (M, dim)
        else:
            initial_simplex_jitter = np.broadcast_to(self.initial_jitter, (1, dim))    # (1, dim)
            assert initial_simplex_jitter.shape == (1, dim)

        curr_simplex = np.expand_dims(initial_points, axis=1)  # (M, 1, dim)
        curr_simplex = np.tile(curr_simplex, (1, dim+1, 1))    # (M, dim+1, dim)
        for k in range(1, dim+1):
            curr_simplex[:, k, k-1] += initial_simplex_jitter[:, k-1]

        curr_loss = func(curr_simplex)  # (M, dim+1)

        return _NMAlgorithmState(simplex=curr_simplex, loss=curr_loss)

    def _update_state(self, state: _NMAlgorithmState) -> _NMAlgorithmState:
        # sort the fitnesses
        sort_indices = np.argsort(state.loss, axis=1)
        new_simplex = np.take_along_axis(sort_indices, state.simplex, 1)
        new_loss = np.take_along_axis(sort_indices, state.loss, 1)

        new_coord_centroid = np.mean(state.simplex[:, :-1])  # (n_nci, dim)
        new_coord_worst = state.simplex[:, -1]                  # (n_nci, dim)
        new_coord_best = state.simplex[:, 0]                    # (n_nci, dim)
        new_coord_secondworst = state.simplex[:, -2]                  # (n_nci, dim)

        new_loss_worst = state.loss[:, -1]                  # (n_nci,)
        new_loss_best = state.loss[:, 0]                    # (n_nci,)
        new_loss_secondworst = state.loss[:, -2]                  # (n_nci,)

        new_state = _NMAlgorithmState(simplex=new_simplex,
                                      loss=new_loss,
                                      coord_best=new_coord_best,
                                      coord_worst=new_coord_worst,
                                      coord_secondworst=new_coord_secondworst,
                                      coord_centroid=new_coord_centroid,
                                      loss_worst=new_loss_worst,
                                      loss_best=new_loss_best,
                                      loss_secondworst=new_loss_secondworst,

                                      # Don't update reflection here
                                      coord_reflected=state.coord_reflected,
                                      loss_reflected=state.loss_reflected)
        return new_state

    def _calculate_reflection(self, state: _NMAlgorithmState, func, mask):
        alpha = 1.0
        curr_centroid = state.coord_centroid[mask]
        curr_worst = state.coord_worst[mask]

        coord_reflected = curr_centroid + alpha*(curr_centroid - curr_worst)
        loss_reflected = func(coord_reflected)

        if state.coord_reflected is None:
            assert np.all(mask)
            new_coord_reflected = coord_reflected
            new_loss_reflected = loss_reflected
        else:
            new_coord_reflected = state.coord_reflected.copy()
            new_loss_reflected = state.loss_reflected.copy()
            new_coord_reflected[mask] = coord_reflected
            new_loss_reflected[mask] = loss_reflected

        new_state = state._replace(coord_reflected=new_coord_reflected,
                                   loss_reflected=new_loss_reflected)
        return new_state

    def _validate_reflection(self, state: _NMAlgorithmState):
        to_reflect = (state.loss_reflected < state.loss_secondworst) & (state.loss_reflected >= state.loss_best)
        to_contract = (state.loss_reflected >= state.loss_secondworst)
        to_expand = (state.loss_reflected < state.loss_best)

        to_outer_contract = to_contract & (state.loss_reflected < state.loss_worst)
        to_inner_contract = state.loss_reflected >= state.loss_worst

        assert np.all((to_inner_contract | to_inner_contract) == to_contract)

        return_masks = (to_reflect, to_expand, to_inner_contract, to_outer_contract)

        # check for no common elements
        assert not any(any(np.any(x == y) for x in return_masks if x is not y) for y in return_masks)
        # check union covers all elements
        assert np.all(np.any(return_masks, axis=0))

        return to_reflect, to_expand, to_inner_contract, to_outer_contract

    def _perform_expansion(self, state: _NMAlgorithmState, func, mask):
        gamma = 2.0
        coord_centroid = state.coord_centroid[mask]
        coord_reflected = state.coord_reflected[mask]
        loss_reflected = state.loss_reflected[mask]

        coord_expanded = coord_centroid + gamma*(coord_reflected - coord_centroid)
        loss_expanded = func(coord_expanded)

        expanded_is_better = loss_expanded < loss_reflected
        coord_expanded = coord_expanded[expanded_is_better]
        loss_expanded = loss_expanded[expanded_is_better]

        return_mask = mask.copy()
        return_mask[mask] &= expanded_is_better

        return coord_expanded, loss_expanded, return_mask

    def _perform_outer_contraction(self, state: _NMAlgorithmState, func, mask):
        # reflection is better than current worst

        beta = 0.5
        coord_centroid = state.coord_centroid[mask]
        coord_reflected = state.coord_reflected[mask]
        loss_reflected = state.loss_reflected[mask]

        coord_contracted = coord_centroid + beta*(coord_reflected - coord_centroid)
        loss_contracted = func(coord_contracted)

        contracted_is_better = loss_contracted <= loss_reflected
        coord_contracted = coord_contracted[contracted_is_better]
        loss_contracted = loss_contracted[contracted_is_better]

        return_mask = mask.copy()
        return_mask[mask] &= contracted_is_better

        return coord_contracted, loss_contracted, return_mask

    def _perform_inner_contraction(self, state: _NMAlgorithmState, func, mask):
        # reflection is worse than current worst

        beta = 0.5
        coord_centroid = state.coord_centroid[mask]
        coord_worst = state.coord_worst[mask]
        loss_worst = state.loss_worst[mask]

        coord_contracted = coord_centroid + beta*(coord_worst - coord_centroid)
        loss_contracted = func(coord_contracted)

        contracted_is_better = loss_contracted < loss_worst
        coord_contracted = coord_contracted[contracted_is_better]
        loss_contracted = loss_contracted[contracted_is_better]

        return_mask = mask.copy()
        return_mask[mask] &= contracted_is_better

        return coord_contracted, loss_contracted, return_mask

    def _perform_shrink(self, state: _NMAlgorithmState, func, mask):
        # reflection is worse than current worst

        delta = 0.5

        coord_rest = state.simplex[mask, 1:]  # (nmask, dim (points in simplex), dim)
        coord_best = state.coord_best[mask].expand_dims(1)  # (nmask, 1, dim)

        new_simplex_coords = coord_best + delta*(coord_rest - coord_best)  # (nmask, dim (points in simplex), dim)
        new_simplex_losses = func(new_simplex_coords)  # (nmask, dim (points in simplex))
        return new_simplex_coords, new_simplex_losses


    def minimize(self, func, initial_points):

        initial_points = np.asarray(initial_points)

        dim = self.input_dim
        M = len(initial_points)
        assert initial_points.shape == (M, dim), \
            (f"Expected initial_points of shape {(M, dim)}, instead got {initial_points.shape}")

        def func_wrapper(x):
            """wrapper for arbitrary dimensional x"""
            x_flat = np.reshape(x, (-1, dim))
            func_flat = func(x_flat)
            return np.reshape(func_flat, x.shape[:-1])

        state = self._get_initial_state(func_wrapper, initial_points)
        inter_coords = np.zeros((self.max_steps+1, M, dim))
        inter_losses = np.zeros((self.max_steps+1, M))
        n_steps = np.zeros((M,))

        non_converged_mask = np.ones((M,), dtype=bool)
        for i in range(self.max_steps):
            # sort the fitnesses, calculate best, second worst, worst, and centroid
            state = self._update_state(state)

            if self.return_intermediates:
                inter_coords[i] = state.simplex[:, 0]
                inter_losses[i] = state.loss[:, 0]
                n_steps += non_converged_mask.astype(np.int_)

            # calculate reflection and function ()
            state = self._calculate_reflection(state, non_converged_mask)
            to_reflect, to_expand, to_inner_contract, to_outer_contract = self._validate_reflection(state)

            expanded_coords, expanded_loss, expanded_success = self._perform_expansion(state, func_wrapper, to_expand & non_converged_mask)
            outer_contracted_coords, outer_contracted_loss, outer_contracted_success = self._perform_outer_contraction(state, func_wrapper, to_outer_contract & non_converged_mask)
            inner_contracted_coords, inner_contracted_loss, inner_contracted_success = self._perform_inner_contraction(state, func_wrapper, to_inner_contract & non_converged_mask)

            # since expansion failures are still updated by the reflection point
            to_update_with_reflected = to_reflect | (to_expand & ~expanded_success)

            # All points that don't qualify for reflection, expansion, and which have failed contraction
            to_perform_shrink = non_converged_mask & ~(to_reflect | to_expand | outer_contracted_success | inner_contracted_success)
            
            if self.perform_shrink:
                shrunk_simplex_coords, shrunk_simplex_losses = self._perform_shrink(state, func_wrapper, to_perform_shrink)
            else:
                # all those assigned to shrinking have converged
                non_converged_mask &= ~to_perform_shrink

            # replace worst element of simplex
            # in-place edit is safe due to the simplex being freshly created in _get_initial_state and _update_state
            state.simplex[to_update_with_reflected, -1] = state.coord_reflected[to_update_with_reflected]
            state.loss[to_update_with_reflected, -1] = state.loss_reflected[to_update_with_reflected]

            state.simplex[to_perform_shrink, 1:] = shrunk_simplex_coords
            state.loss[to_perform_shrink, 1:] = shrunk_simplex_losses

        # Perform final state update
        state = self._update_state(state)
        inter_coords[self.max_steps] = state.simplex[:, 0]
        inter_losses[self.max_steps] = state.loss[:, 0]
        inter_coords = np.transpose(inter_coords, (1, 0, 2))
        inter_losses = np.transpose(inter_losses, (1, 0))

        if self.return_intermediates:
            return_val = NelderMeadResult(x=state.coord_best, loss=state.loss_best, 
                                          intermediate=NelderMeadIntermediates(coords=inter_coords,
                                                                               losses=inter_losses,
                                                                               n_steps=n_steps))
        
        return return_val