from __future__ import print_function, division, absolute_import

import GPy
import numpy as np
import matplotlib as mpl
import stage_opt
import matplotlib.pyplot as plt
from utilities import (plot_2d_gp, plot_3d_gp, plot_contour_gp,
                        linearly_spaced_combinations, sample_gp_function)

mpl.rcParams['figure.figsize'] = (20.0, 10.0)
mpl.rcParams['font.size'] = 20
mpl.rcParams['lines.markersize'] = 20

# 设置输入安全限制数目n，则gp总数为n+1，第一个是效用函数
safe_dim = 1

# 设置需查找的区间
bounds = [(-5., 5.)]

# 设置测量不确定度，即可认为是误差方差
noise_var = []

# 设置kernel
kernel = []
for i in range(0, safe_dim + 1):
    kernel.append(GPy.kern.RBF(input_dim=len(bounds), variance=2, lengthscale=1.0, ARD=True))
    noise_var.append(0.05 ** 2)

# 确定样本集内点数
n_D = 1000

# 初始化样本集D
D = linearly_spaced_combinations(bounds, n_D)

# 初始化最初的安全点safe_point \in S
x0 = np.zeros((1, len(bounds)))

# 初始化安全阈值
safe_h = [-np.inf, 0., 0.]


# 初始化安全函数阈值，须满足在初始点处安全
def sample_safe_fun():
    fun = np.hstack(sample_gp_function(kernel[0], bounds, noise_var[0], 100))
    for i in range(0, safe_dim):
        while True:
            fun_temp = sample_gp_function(kernel[i + 1], bounds, noise_var[i + 1], 100)
            if fun_temp(x0[0][0], noise=False) > (safe_h[i + 1] + 0.5):
                break
        fun.hstack(fun, fun_temp)
    return fun


fun = sample_safe_fun()

# 根据初始点建立高斯统计模型
y0 = fun(x0)
gp = []
for i in range(0, safe_dim + 1):
    gp.append(GPy.models.GPRegression(x0, y0[:, i, None], kernel[i], noise_var[i]))


opt = stage_opt.StageOpt(gp, D, safe_h, lipschitz=None, threshold=0.5)
