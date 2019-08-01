"""
Classes that implement SafeOpt.

Authors: - Felix Berkenkamp (befelix at inf dot ethz dot ch)
         - Nicolas Carion (carion dot nicolas at gmail dot com)
"""

from __future__ import print_function, absolute_import, division

from collections import Sequence
from functools import partial

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import expit
from scipy.stats import norm
from builtins import range

from utilities import (plot_2d_gp, plot_3d_gp, plot_contour_gp,
                       linearly_spaced_combinations)
from swarm import SwarmOptimization

import logging

__all__ = ['StageOpt']


class GaussianProcessOptimization(object):
    """
    Base class for GP optimization.

    Handles common functionality.

    Parameters
    ----------
    gp: GPy Gaussian process
    fmin : float or list of floats
        Safety threshold for the function value. If multiple safety constraints
        are used this can also be a list of floats (the first one is always
        the one for the values, can be set to None if not wanted).
    beta: float or callable
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float or list of floats
        The algorithm will not try to expand any points that are below this
        threshold. This makes the algorithm stop expanding points eventually.
        If a list, this represents the stopping criterion for all the gps.
        This ignores the scaling factor.
    scaling: list of floats or "auto"
        A list used to scale the GP uncertainties to compensate for
        different input sizes. This should be set to the maximal variance of
        each kernel. You should probably leave this to "auto" unless your
        kernel is non-stationary.
    """

    def __init__(self, gp, fmin, beta=2, num_contexts=0, threshold=0,
                 scaling='auto'):
        """Initialization, see `GaussianProcessOptimization`."""
        super(GaussianProcessOptimization, self).__init__()

        if isinstance(gp, list):
            self.gps = gp
        else:
            self.gps = [gp]
        self.gp = self.gps[0]

        self.fmin = fmin
        if not isinstance(self.fmin, list):
            self.fmin = [self.fmin] * len(self.gps)
        self.fmin = np.atleast_1d(np.asarray(self.fmin).squeeze())

        if hasattr(beta, '__call__'):
            # Beta is a function of t
            self.beta = beta
        else:
            # Assume that beta is a constant
            self.beta = lambda t: beta

        if scaling == 'auto':
            dummy_point = np.zeros((1, self.gps[0].input_dim))
            self.scaling = [gpm.kern.Kdiag(dummy_point)[0] for gpm in self.gps]
            self.scaling = np.sqrt(np.asarray(self.scaling))
        else:
            self.scaling = np.asarray(scaling)
            if self.scaling.shape[0] != len(self.gps):
                raise ValueError("The number of scaling values should be "
                                 "equal to the number of GPs")

        self.threshold = threshold
        self._parameter_set = None
        self.bounds = None
        self.num_samples = 0
        self.num_contexts = num_contexts

        self._x = None
        self._y = None
        self._get_initial_xy()

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def data(self):
        """Return the data within the GP models."""
        return self._x, self._y

    @property
    def t(self):
        """Return the time step (number of measurements)."""
        return self._x.shape[0]

    def _get_initial_xy(self):
        """Get the initial x/y data from the GPs."""
        self._x = self.gp.X
        y = [self.gp.Y]

        for gp in self.gps[1:]:
            if np.allclose(self._x, gp.X):
                y.append(gp.Y)
            else:
                raise NotImplemented('The GPs have different measurements.')

        self._y = np.concatenate(y, axis=1)

    def plot(self, n_samples, safe_region = None, axis=None, figure=None, plot_3d=False,
             **kwargs):
        """
        Plot the current state of the optimization.

        Parameters
        ----------
        n_samples: int
            How many samples to use for plotting
        axis: matplotlib axis
            The axis on which to draw (does not get cleared first)
        figure: matplotlib figure
            Ignored if axis is already defined
        plot_3d: boolean
            If set to true shows a 3D plot for 2 dimensional data
        """
        # Fix contexts to their current values
        if self.num_contexts > 0 and 'fixed_inputs' not in kwargs:
            kwargs.update(fixed_inputs=self.context_fixed_inputs)

        true_input_dim = self.gp.kern.input_dim - self.num_contexts
        if true_input_dim == 1 or plot_3d:
            inputs = np.zeros((n_samples ** true_input_dim, self.gp.input_dim))
            inputs[:, :true_input_dim] = linearly_spaced_combinations(
                self.bounds[:true_input_dim],
                n_samples)

        if not isinstance(n_samples, Sequence):
            n_samples = [n_samples] * len(self.bounds)


        axes = []
        if self.gp.input_dim - self.num_contexts == 1:
            # 2D plots with uncertainty
            for gp, fmin in zip(self.gps, self.fmin):
                temp_sr = safe_region
                if fmin == -np.inf:
                    fmin = None
                    temp_sr = None
                ax = plot_2d_gp(gp, inputs, figure=figure, axis=axis, safe_region = safe_region,
                                fmin=fmin, **kwargs)
                axes.append(ax)
        else:
            if plot_3d:
                for gp in self.gps:
                    plot_3d_gp(gp, inputs, figure=figure, axis=axis, **kwargs)
            else:
                for gp in self.gps:
                    plot_contour_gp(gp,
                                    [np.linspace(self.bounds[0][0],
                                                 self.bounds[0][1],
                                                 n_samples[0]),
                                     np.linspace(self.bounds[1][0],
                                                 self.bounds[1][1],
                                                 n_samples[1])],
                                    figure=figure,
                                    axis=axis)

    def _add_context(self, x, context):
        """Add the context to a vector.

        Parameters
        ----------
        x : ndarray
        context : ndarray

        Returns
        -------
        x_extended : ndarray
        """
        context = np.atleast_2d(context)
        num_contexts = context.shape[1]

        x2 = np.empty((x.shape[0], x.shape[1] + num_contexts), dtype=float)
        x2[:, :x.shape[1]] = x
        x2[:, x.shape[1]:] = context
        return x2

    def _add_data_point(self, gp, x, y, context=None):
        """Add a data point to a particular GP.

        This should only be called on its own if you know what you're doing.
        This does not update the global data stores self.x and self.y.

        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        context: array_like
            The context(s) used for the data points
        gp: instance of GPy.model.GPRegression
            If specified, determines the GP to which we add the data point
            to. Note that this should only be used if that data point is going
            to be removed again.
        """
        if context is not None:
            x = self._add_context(x, context)

        gp.set_XY(np.vstack([gp.X, x]),
                  np.vstack([gp.Y, y]))

    def add_new_data_point(self, x, y, context=None):
        """
        Add a new function observation to the GPs.

        Parameters
        ----------
        x: 2d-array
        y: 2d-array
        context: array_like
            The context(s) used for the data points.
        """
        x = np.atleast_2d(x)
        y = np.atleast_2d(y)

        if self.num_contexts:
            x = self._add_context(x, context)

        for i, gp in enumerate(self.gps):
            not_nan = ~np.isnan(y[:, i])
            if np.any(not_nan):
                # Add data to GP (context already included in x)
                self._add_data_point(gp, x[not_nan, :], y[not_nan, [i]])

        # Update global data stores
        self._x = np.concatenate((self._x, x), axis=0)
        self._y = np.concatenate((self._y, y), axis=0)

    def _remove_last_data_point(self, gp):
        """Remove the last data point of a specific GP.

        This does not update global data stores, self.x and self.y.

        Parameters
        ----------
            gp: Instance of GPy.models.GPRegression
                The gp that the last data point should be removed from
        """
        gp.set_XY(gp.X[:-1, :], gp.Y[:-1, :])

    def remove_last_data_point(self):
        """Remove the data point that was last added to the GP."""
        last_y = self._y[-1]

        for gp, yi in zip(self.gps, last_y):
            if not np.isnan(yi):
                gp.set_XY(gp.X[:-1, :], gp.Y[:-1, :])

        self._x = self._x[:-1, :]
        self._y = self._y[:-1, :]


class StageOpt(GaussianProcessOptimization):
    """A class for Safe Bayesian Optimization.

    实现StageOpt算法，使用高斯模型用来保证参数集合内元素安全。
    StageOpt算法在不同纬度上评价安全函数(safe functions)和效用函数(utility function)
    在本方法中默认效用函数有且仅有一个(R^n -> R^1)，安全函数至少有一个(R^n -> R^1)
    效用函数和安全函数不同

    step_1:扩展安全域
    step_2:寻找安全域内最优解

    Parameters
    ----------
    gp: 使用GPy库构造的高斯过程
        该高斯过程是使用初始化给出的安全点集构造的，
        如果输入为一个GP list的话，取第一个GP作为拟合得到的效用函数，其他GP都是拟合的安全限制函数
    parameter_set: 2d-array 2维数组
        参数列表，我的理解是样本全集D
    fmin: list of floats 浮点数列表
        安全函数的最小值限制。
        列表的第一个元素是效用函数的最小值限制，如果不需要，可以设置为None
    lipschitz: list of floats
        本类使用的Lipschitz常数，
        如果为None则GP可信度直接被使用
    beta: float or callable
        一个常数值或者一个可被计算的函数
        A constant or a function of the time step that scales the confidence
        interval of the acquisition function.
    threshold: float or list of floats
        算法将不会扩展在threhold之下的点，该参数决定了算法何时停止扩张。
        如果输入为一个列表，则反映了所有GPs的停止扩张标准
        该参数优先级高于scaling参数
    scaling: list of floats or "auto"
        该列表用于衡量GP不确定度在不同输入大小上的损失。
        被设置为每个kernel的最大方差
        你应该尽可能设置其为"auto"，除非kernel是不合适的
    bound_index:safe region
        已经可以被确定的安全区间

    Examples
    --------
    # >>> from safeopt import SafeOpt
    # >>> from safeopt import linearly_spaced_combinations
    # >>> import GPy
    # >>> import numpy as np

    Define a Gaussian process prior over the performance

    # >>> x = np.array([[0.]])
    # >>> y = np.array([[1.]])
    # >>> gp = GPy.models.GPRegression(x, y, noise_var=0.01**2)
    #
    # >>> bounds = [[-1., 1.]]
    # >>> parameter_set = linearly_spaced_combinations([[-1., 1.]],
    # ...                                              num_samples=100)

    Initialize the Bayesian optimization and get new parameters to evaluate

    # >>> opt = SafeOpt(gp, parameter_set, fmin=[0.])
    # >>> next_parameters = opt.optimize()

    Add a new data point with the parameters and the performance to the GP. The
    performance has normally be determined through an external function call.

    # >>> performance = np.array([[1.]])
    # >>> opt.add_new_data_point(next_parameters, performance)
    """

    def __init__(self, gp, Seed, parameter_set, fmin, lipschitz=None, beta=2,
                 num_contexts=0, threshold=0, scaling='auto'):
        """Initialization, see `SafeOpt`."""
        super(StageOpt, self).__init__(gp,
                                       fmin=fmin,
                                       beta=beta,
                                       num_contexts=num_contexts,
                                       threshold=threshold,
                                       scaling=scaling)

        # 使用bound用来表示可确定的安全区间，初始时，安全区间的范围为空
        self.bound = np.array([parameter_set[Seed][0], parameter_set[Seed][0]])
        self.bound_index = np.array([Seed, Seed])

        if self.num_contexts > 0:
            context_shape = (parameter_set.shape[0], self.num_contexts)
            self.inputs = np.hstack((parameter_set,
                                     np.zeros(context_shape,
                                              dtype=parameter_set.dtype)))
            self.parameter_set = self.inputs[:, :-self.num_contexts]
        else:
            self.inputs = self.parameter_set = parameter_set

        self.liptschitz = lipschitz

        if self.liptschitz is not None:
            if not isinstance(self.liptschitz, list):
                self.liptschitz = [self.liptschitz] * len(self.gps)
            self.liptschitz = np.atleast_1d(
                np.asarray(self.liptschitz).squeeze())

        # Value intervals
        self.Q = np.empty((self.inputs.shape[0], 2 * len(self.gps)),
                          dtype=np.float)

        # Safe set
        self.S = np.zeros(self.inputs.shape[0], dtype=np.bool)
        self.S[Seed] = True

        # Switch to use confidence intervals for safety
        if lipschitz is None:
            self._use_lipschitz = False
        else:
            self._use_lipschitz = True

        # Set of expanders and maximizers
        self.G = self.S.copy()
        self.M = self.S.copy()

    @property
    def use_lipschitz(self):
        """
        Boolean that determines whether to use the Lipschitz constant.

        By default this is set to False, which means the adapted SafeOpt
        algorithm is used, that uses the GP confidence intervals directly.
        If set to True, the `self.lipschitz` parameter is used to compute
        the safe and expanders sets.
        """
        return self._use_lipschitz

    @use_lipschitz.setter
    def use_lipschitz(self, value):
        if value and self.liptschitz is None:
            raise ValueError('Lipschitz constant not defined')
        self._use_lipschitz = value

    @property
    def parameter_set(self):
        """Discrete parameter samples for Bayesian optimization."""
        return self._parameter_set

    @parameter_set.setter
    def parameter_set(self, parameter_set):
        self._parameter_set = parameter_set

        # Plotting bounds (min, max value
        self.bounds = list(zip(np.min(self._parameter_set, axis=0),
                               np.max(self._parameter_set, axis=0)))
        self.num_samples = [len(np.unique(self._parameter_set[:, i]))
                            for i in range(self._parameter_set.shape[1])]

    @property
    def context_fixed_inputs(self):
        """Return the fixed inputs for the current context."""
        n = self.gp.input_dim - 1
        nc = self.num_contexts
        if nc > 0:
            contexts = self.inputs[0, -self.num_contexts:]
            return list(zip(range(n, n - nc, -1), contexts))

    @property
    def context(self):
        """Return the current context variables."""
        if self.num_contexts:
            return self.inputs[0, -self.num_contexts:]

    @context.setter
    def context(self, context):
        """Set the current context and update confidence intervals.

        Parameters
        ----------
        context: ndarray
            New context that should be applied to the input parameters
        """
        if self.num_contexts:
            if context is None:
                raise ValueError('Need to provide value for context.')
            self.inputs[:, -self.num_contexts:] = context

    # 对D中所有元素计算计算Q_t^i
    def update_confidence_intervals(self, context=None):
        """Recompute the confidence intervals form the GP.

        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        """
        beta = self.beta(self.t)

        # Update context to current setting
        self.context = context

        # Iterate over all functions
        for i in range(len(self.gps)):
            # Evaluate acquisition function
            mean, var = self.gps[i].predict_noiseless(self.inputs)

            mean = mean.squeeze()
            std_dev = np.sqrt(var.squeeze())

            # Update confidence intervals
            self.Q[:, 2 * i] = mean - beta * std_dev
            self.Q[:, 2 * i + 1] = mean + beta * std_dev

    # 计算安全集合S_t
    # 重写计算安全集合函数，原函数未使用d(x,x')在重写函数中使用
    # 更新安全区间bound
    def compute_safe_set(self):
        """Compute only the safe set based on the current confidence bounds."""
        # Update safe set
        for i in range(len(self.S)):
            if (self.inputs[i] < self.bound[0]) or (self.inputs[i] > self.bound[1]):
                d = min(abs(self.inputs[i] - self.bound[:]))
                # print(d)
                self.S[i] = np.all(self.Q[i, ::2] > self.fmin + d * self.liptschitz)
                if self.S[i]:
                    self.bound[0] = min(self.bound[0], self.inputs[i])
                    self.bound[1] = max(self.bound[1], self.inputs[i])
                    self.bound_index[0] = min(self.bound_index[0], i)
                    self.bound_index[0] = max(self.bound_index[0], i)
            else:
                self.S[i] = True

        # self.S[:] = np.all(self.Q[:, ::2] > self.fmin, axis=1)

    # 计算G_t
    def compute_sets(self):
        """
        Compute the safe set of points, based on current confidence bounds.

        Parameters
        ----------
        context: ndarray
            Array that contains the context used to compute the sets
        full_sets: boolean
            Whether to compute the full set of expanders or whether to omit
            computations that are not relevant for running SafeOpt
            (This option is only useful for plotting purposes)
        """

        # Update safe set
        self.compute_safe_set()

        if not np.any(self.S):
            self.G[:] = False
            return

        # Optimistic set of possible expanders
        u = self.Q[:, 1::2]

        self.G[:] = False

        s = self.S

        # set of safe expanders
        G_safe = np.zeros(np.count_nonzero(s), dtype=np.bool)
        sort_index = range(len(G_safe))

        for index in sort_index:
            # Distance between current index point and all other unsafe
            # points
            d = cdist(self.inputs[s, :][[index], :],
                      self.inputs[~self.S, :])

            # Check if expander for all GPs
            for i in range(len(self.gps)):
                # Skip evaluation if 'no' safety constraint
                if self.fmin[i] == -np.inf:
                    continue
                # Safety: u - L * d >= fmin
                G_safe[index] = \
                    np.any(u[s, i][index] - self.liptschitz[i] * d >=
                           self.fmin[i])
                # Stop evaluating if not expander according to one
                # safety constraint
                if not G_safe[index]:
                    break

        # Update safe set (if full_sets is False this is at most one point
        self.G[s] = G_safe

    def get_expand_point(self):
        l = self.Q[:, ::2]
        u = self.Q[:, 1::2]

        self.G[self.G] = np.any((u[self.G, :] - l[self.G, :]) >= self.threshold)

        if not np.any(self.G):
            return self.get_optimize_point()

        temp_max = 0
        temp_index = 0
        for i in range(len(self.G)):
            if self.G[i]:
                temp_w = np.max(u[i] - l[i])
                if temp_w > temp_max:
                    temp_max = temp_w
                    temp_index = i
        return self.inputs[temp_index,:]

    def get_optimize_point(self):
        if not np.any(self.S):
            raise EnvironmentError('There are no safe points to evaluate.')

        max_id = np.argmax(self.Q[self.S, 1])
        x = self.inputs[self.S, :][max_id, :]
        return x


    def expansion(self, context=None):
        self.update_confidence_intervals(context=context)

        # 更新集合内元素
        self.compute_sets()

        return self.get_expand_point()

    def optimize(self, context=None, ucb=False):
        """Run Safe Bayesian optimization and get the next parameters.

        Parameters
        ----------
        context: ndarray
            A vector containing the current context
        ucb: bool
            If True the safe-ucb criteria is used instead.

        Returns
        -------
        x: np.array
            The next parameters that should be evaluated.
        """
        # Update confidence intervals based on current estimate
        self.update_confidence_intervals(context=context)

        self.compute_safe_set()

        return self.get_optimize_point()

    def get_maximum(self, context=None):
        """
        Return the current estimate for the maximum.

        Parameters
        ----------
        context: ndarray
            A vector containing the current context

        Returns
        -------
        x - ndarray
            Location of the maximum
        y - 0darray
            Maximum value

        Notes
        -----
        Uses the current context and confidence intervals!
        Run update_confidence_intervals first if you recently added a new data
        point.
        """
        self.update_confidence_intervals(context=context)

        # Compute the safe set (that's cheap anyways)
        self.compute_safe_set()

        # Return nothing if there are no safe points
        if not np.any(self.S):
            return None

        l = self.Q[self.S, 0]

        max_id = np.argmax(l)
        return (self.inputs[self.S, :][max_id, :-self.num_contexts or None],
                l[max_id])
