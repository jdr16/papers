# [Sources of suboptimality in a minimalistic explore–exploit task](https://www.nature.com/articles/s41562-018-0526-x)

- explore-exploite decisions、由exploring开始和至多一次的exploitation结束的贯续决策任务、threshold、'sequence-level'

- 研究explore-exploit问题面临的难题：
  - 效用函数未知
  - 人类需要记住之前做的选择和对应回报
  - exploit对应的回报存在不确定性
  - 环境可变

## Main Idea

- 背景描述：
  - 参与者对任务环境有充分了解
  - exploration得到的值在可选样本集中随机抽取
  - $r$: reward 可选样本集对应reward满足1-5的高斯分布
  - $T$: trail kength5-10次选择
  - $a$: action
    - $a=1$: exploration：随机选择未选过的；
    - $a=0$: exploitation：选择过去选择的最好的
  - $t_{left}$: 剩余时间/选择次数
  - $(r^*,t_{left})$: given state，状态描述
  - $Q(r^*,t_{left},a), ~~ a=0,1$: 状态$(r^*,t_{left})$下选择对应的回报
    - 当$r^*$较小，$t_{left}$较大时，$a=1$的可能性更大，换言之更多exploration

- 目的：所有选择得到的总分最高

## Experiment

- 参与者：
  - 实验室被试：49人
  - 线上被试：143人

- 搜索
  - 实验室：$T=180$,(1170 choices)
  - 线上：$T=60$,(490 choices)

- 其他
  - 实验室被试者在实验结束之后被要求总结并写下自己的策略

- 结果：
  - $t_{left}\uparrow,~P(a=1)\downarrow$
  - $max(Q_{past})\uparrow,~P(a=1)\downarrow$
  - $t_{left}$ 不变 $T\downarrow, P(a=1)\uparrow$

## Model

- ***Model 1: Opt Model***
  - 考虑 softmax noise
  - 设置
      $$P(a=1)=f(\beta_0+\beta\Delta Q(r^*,t_{left}))$$

    - $\Delta Q(r^*,t_{left})\equiv  Q(r^*,t_{left};1)-Q(r^*,t_{left};0)$
    - $f(x)=\frac{1}{1+e^{-x}}$
    - $\beta$ inverse temperature
    - $\beta_0$ 模型自带的bias
  - 结果
    - 没有得到很好的拟合效果
    - 未考虑$T-trail length$的影响
    - **决策不仅仅与 softmax noise 和 bias 有关**
-***Model 2: Num Model***
  - heuristic strategies & time-dependent threshold
  - 设置
     $$P(a=1)=f(\beta(\theta - r^*))$$

    - $\theta(t_left)$: 阈值决定函数，$\theta = kt_{left}+b$($k,b$为模型参数)
  - 结果
    - 在exploration部分表现更好
  - 原因
    - 未考虑到$T$的影响

- ***Model 3: Prob Model***
  - $\theta(t_left,T)$
  - 设置
      $$P(a=1)=f(\beta(\theta - r^*))$$
    - $\theta(t_left,T)$: 阈值决定函数，$\theta = kt_{left}/T+b$($k,b$为模型
  - 结果
    - 在explore-exploit问题中，相对进度比绝对进度更重要

- ***Model 4: Prop-V & Num-V Model***
  - **Sequence-level variability**
    - 在每一次决策之前，都会更新对每一可能决策的threshold
    - *引入同一trail内不同决策间的依赖*
  - 设置
      $$P(a=1)=f(\beta(\theta+\eta-r^*))$$
    - $\eta \sim\mathcal{N}(0,\sigma^2)$
  - 结果：
    - Prop-V Model 对数据的拟合效果最好

## Result

### Git Repository

