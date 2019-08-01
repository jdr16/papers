# Selection Bias Explorations and Debias Methods for Natural Language Sentence Matching Datasets

句对匹配任务中的样本选择偏差与去偏方法

## Main Idea

- 证明NLSM datasets中存在选择偏差(selection bias)

- 选择偏差对深度神经网络(DNN)的训练结果有影响

- 提出改善方案

## Assume

- $\mathcal{X,Y,L,S}$张成**不存在特征泄露**(leakage-neutral)的集合$\mathcal{D}$。
  - $\mathcal{X}$：语义特征空间(semantic feature space)
  - $\mathcal{Y}$：由0/1构成的语义标签空间(binary semantic label space)
  - $\mathcal{L}$：取样策略空间(sampling strategy feature space)
  - $\mathcal{S}$：0/1构成的取样意图空间(binary sampling intention space)，*当$\mathcal{S} = 1$时，我们认为数据集提供者希望选取一个正样本(positive sample)*

- 样本$(x,y,l,s)$是独立地在$\mathcal{D}$中抽取的，只有当$s=y$ -- **标签符合取样意图**(label matches the sampling intention) 时该样本才会被选择进入实际样本集$\mathcal{\hat{D}}$。这也就造成了样本集的偏差。

- 取样策略与语义标签独立，$P(Y|L)=P(Y)$
- 取样策略空间、取样意图空间与语义特征空间、语义标签空间独立：$P(S|X,Y,L)=P(S|L)$

## Method

- 已知先验概率$P(Y=1)$
- 样本集合中估计$P_{\mathcal{\hat{D}}}(Y=1|l)$
- 可推出如下公式，计算得$P(S=1|l)$

  $$P(S=1|l) = \frac{P(Y=0)P_{\mathcal{\hat{D}}}(Y=1|l)}{P(Y=0)P_{\mathcal{\hat{D}}}(Y=1|l) + P(Y=1)P_{\mathcal{\hat{D}}}(Y=0|l)}$$

- 得到样本权重$w$

$$w=\frac{1}{P(S=y|l)}$$

## Experients

### Git Repository
[Selection Bias Explorations and Debias Methods for Natural Language Sentence Matching Datasets](https://github.com/arthua196/Leakage-Neutral-Learning-for-QuoraQP.git)
