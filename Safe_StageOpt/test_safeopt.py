from __future__ import print_function, division, absolute_import

import GPy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import safeopt

def test_safeopt_fun():
    mpl.rcParams['figure.figsize'] = (20.0, 10.0)
    mpl.rcParams['font.size'] = 20
    mpl.rcParams['lines.markersize'] = 20

    # Measurement noise
    noise_var = 0.05 ** 2
    noise_var2 = 1e-5
    noise_var3 = 1e-3

    # Bounds on the inputs variable
    bounds = [(-10., 10.)]

    # Define Kernel
    kernel = GPy.kern.RBF(input_dim=len(bounds), variance=2., lengthscale=1.0, ARD=True)
    kernel2 = kernel.copy()
    kernel3 = kernel.copy()

    # set of parameters
    parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)

    # Initial safe point
    x0 = np.zeros((1, len(bounds)))


    # Generate function with safe initial point at x=0
    def sample_safe_fun():
        fun = safeopt.sample_gp_function(kernel, bounds, noise_var, 100)
        while True:
            fun2 = safeopt.sample_gp_function(kernel2, bounds, noise_var2, 100)
            if fun2(0, noise=False) > 1:
                break
        while True:
            fun3 = safeopt.sample_gp_function(kernel3, bounds, noise_var3, 100)
            if fun3(0, noise=False) > 1:
                break

        def combined_fun(x, noise=True):
            return np.hstack([fun(x, noise), fun2(x, noise), fun3(x, noise)])

        return combined_fun

    # Define the objective function
    fun = sample_safe_fun()

    # The statistical model of our objective function and safety constraint
    y0 = fun(x0)
    gp = GPy.models.GPRegression(x0, y0[:, 0, None], kernel, noise_var=noise_var)
    gp2 = GPy.models.GPRegression(x0, y0[:, 1, None], kernel2, noise_var=noise_var2)
    gp3 = GPy.models.GPRegression(x0, y0[:, 2, None], kernel2, noise_var=noise_var2)

    # The optimization routine
    # opt = safeopt.SafeOptSwarm([gp, gp2], [-np.inf, 0.], bounds=bounds, threshold=0.2)
    opt = safeopt.SafeOpt([gp, gp2, gp3], parameter_set, [-np.inf, 0., 0.], lipschitz=0.1, threshold=1)


    def plot():
        # Plot the GP
        opt.plot(100)
        # Plot the true function
        y = fun(parameter_set, noise=False)
        for manager, true_y in zip(mpl._pylab_helpers.Gcf.get_all_fig_managers(), y.T):
            figure = manager.canvas.figure
            figure.gca().plot(parameter_set, true_y, color='C2', alpha=0.3)

    while opt.t < 25:
        # Obtain next query point
        x_next = opt.optimize()
        # Get a measurement from the real system
        y_meas = fun(x_next)
        # Add this to the GP model
        opt.add_new_data_point(x_next, y_meas)
        if (y_meas[0][1] < 0) or (y_meas[0][2] < 0):
            print('error')
            return 1
            break

    return 0

times = 0
for i in range(1000):
    times += test_safeopt_fun()

print(times)

