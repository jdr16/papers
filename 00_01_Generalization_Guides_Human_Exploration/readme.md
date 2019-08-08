# [Generalization guides human exploration in vast decision spaces](https://www.nature.com/articles/s41562-018-0467-4)

- 许多人类行为可以被建模为在大量可能行为张成的空间内的搜索问题。本文主要研究的问题是：在相似的选择会带来相似的回报(spatial correlation of rewards)的有限空间条件下，人类如何寻求最优回报。

- 使用到的工具：
  - Gaussian process function learning
  - optimistic upper confidence bound sampling strategy

## Main Idea

- 背景描述：
  - 大量可选策略
  - 回报只有在执行策略之后才可观测得到
  - 时间资源有限(limited time or resources)
  - 已知环境模型
  - 相似选择带来的回报也相似

- 目的： exploration-exploitation dilemma
  - exploration：未知策略(unkown options)
  - exploitation：熟悉策略(familiar options)

- 方法：
  - function learning：使用已知点的观测值预测未知点的值(approximate a global value function over all options)。
  - 基于高斯过程的function learning和UCB采样方式

- 结果：
  - 发现对人类的泛化方式的一种有效估计
  - 发现在支持学习和推理的环境下的传统联想学习(associative learning)的局限性

- 要点：
  - spatially correlated multi-armed bandit as a paradigm 
  - 以高斯过程模型为基础的function learning强有力的建模了人类如何学习和泛化环境结构
  - 强调不确定性

## Experiment

### Experiment 1:

- 参与者人数：n = 81
- 搜索空间：1 x 30， 网格
  - 每个格子都对应一个回报值
  - 格子之间的距离代表选择的相似性(spatially corelation)
  - 回报(reward)与选择相关
- 实验次数：
  - short：5
  - long：10
- 目标：
  - A:平均最大收获--accumulation
  - M:单次最大收获--maximization
- 环境：
  - Smooth：回报与选择相关性强
  - Rough：回报与选择相关性较弱
- 实验假设：
  - function learning 指导搜索过程
  - 参与者在smooth环境中得分高&学习快

- 实验结果：
  - A VS. M: 
    - A组更多的在本地取样，即在熟悉区域进行决策
    - 通常A会获得更高的平均回报
    - A和M在寻找全剧最优解上表现相同
  - S VS. R: 
    - 短时间内得分更高

### Experiment 2:

- **基本与实验1相同，二维空间，只列出不同的实验设置和结果**

- 搜索空间：11 x 11， 网格

- 实验次数：
  - short：20
  - long：40
- 实验结果：
  - A VS. M: 
    - A组更多的在本地取样，即在熟悉区域进行决策
    - A和M在寻找全剧最优解上表现相同
    - **A组并没有表现出明显高于M组的回报值**
  - S VS. R: 
    - 短时间内得分更高
  - 短时间和长时间的实验得到的最终结果相差不大
    - **学习速度快且很快到达峰值**

### Experiment 3:

- **基本与实验2相同，创造更‘自然’的复杂环境，sample之间存在非固定的间接的相关性，只列出不同的实验设置和结果**

- 搜索空间：与实验2相同的121个可选样本， **样本空间关系不固定(don't have fixed spatial correlation)，通过采样获得**
- 实验结果：
  - A VS. M: 
    - A组更多的在本地取样，即在熟悉区域进行决策
    - A和M在寻找全剧最优解上表现相同
    - **A组的平均回报(average reward)明显高于M组**
  - S VS. R: 
    - 短时间内得分更高
  - 短时间和长时间的实验得到的最终结果相差不大
    - 学习速度快且很快到达峰值

## Model Setting

- 通过高斯过程模型对人类function learning 进行建模，通过均值$m(x)$和不确定度$s(x)$指导对人类行为的预测，选择RBF作为核(k)，由下述式子判定
   $$k_{RBF}(x,x')=exp(-\frac{||x-x'||^2}{\lambda})$$
  - 其中$\lambda$为尺度衡量参数，相同距离的两个样本，$\lambda$越大，相关性越小。**本实验中$\lambda = 1$**

- 使用UCB作为采样策略：
    $$UCB(x)=m(x)+\beta s(x)$$
  - 其中$\beta$决定了不确定度对采样的影响，$\beta$越大影响越大。**本实验中$\beta = 0.5$**
  - 特殊情况：
    - $PureExploit(x) = m(x)$，只考虑回报/均值
    - $PureExplore(x) = s(x)$，只考虑不确定性
- 使用参数$\tau$以估计非直接的探索(estimate the amount of unfirected exporation)
  - higher $\tau$ corresponde to more noisy sampling
  - 功能与$\beta$相反
  - ***？？？？？？？？？？？***

- 尝试多种方法描述被试者行为

## Result

- 被试者更好的被function learning模型描述而不是option learning：
  - 被试者更多的使用泛化能力进行学习而非独立学习每次的结果
- 均值预测和不确定性预测对function learning来说都很重要
- **参数:$\beta,\tau$**
  - 参数:$\beta,\tau$的估计值在三次实验中相差不大，且有很强的鲁棒性：
  - 对于参数$\beta$的估计：($\hat{\beta} = 0.51$)，反映了取样过程中对不确定性估计的重要性
  - 对于参数$\tau$的估计：($\hat{\tau} = 0.01$)，说明如果将预测均值和不确定度都考虑在内，被试者的搜索行为与是否选择到过一个很不错的选项更有关

- *Exp 2*：在Smooth和rough两种环境中都被试者都低估了样本间的空间相关性
- *Exp 3*：可能是环境的高度丰富性导致，模型难以通过泛化模型预测出给定样本集合之外的选择
- **参数$\lambda$**
  - 对$\lambda$的较小估计可能不符合人类的认知规律，但会导致更好的表现

### Git Repository

