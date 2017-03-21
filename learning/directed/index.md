---
layout: post
title: 有向模型中的學習
---
现在我们把注意力转到本课程的第三部份也是最后一部分：「学习」。
已知一数据集，调适一个模型使其可以做出我们需要的预测。

图模型有两个部份：图结构、该图导出的因子的参数。这两部份对应两种不同的学习任务：

- **参数学习**，对已知的图结构估计参数
- **结构学习**，估计图结构，即从数据中确定变量彼此之间的关系

我们首先专注于参数学习，然后在后面的章节讲结构学习。

我们考虑有向与无向图中的参数学习。前者允许简单的闭形式解，而后者则可能需要难解的
数字优化技巧。
后面的章节裡我们会讲隐变量模型（包含未观测的变量的模型）以及
贝叶斯学习，一个统计的一般性方法，有时比标准作法更有效。

## 学习任务

Before, we start our discussion of learning, let's first reflect on what it means to fit a model, and what is a desirable objective for this task.

假设定义域有著分布 $$p^* $$。我们的数据集 $$D$$ 包含分布 $$p^* $$ 的 $$m$$ 个样本。标准的假设是这些样本独立同分布(iid)。又已知一类模型 $$M$$，我们的任务是
学习$$M$$裡面一些「好」的模型，对应一个分布 $$p$$。例如我们可以考虑
图结构相同只是CPD表不同的一类贝叶斯网络。

学习的目标是得到一个模型准确再现数据样本的分布 $$p^* $$。
一般情况下这是不可能的，因為有限的数据只能提供真正分布的一个粗略近似，还有计算上的限制。
然而我们还是想要用否种标准选择真正分布 $$p^* $$ 最好的一个近似。

在这裡「最好」是什么意思？这取决于我们的目标：

- 估计概率密度：我们感兴趣的是整个分布（这样我们就能计算任何有需要的条件概率）
- 特定的预测任务：我们使用分布来预测。例如，一封邮件是不是垃圾邮件？
- 结构或知识发现：我们感兴趣的是模型本身。例如，基因之间如何互相影响？

## 最大似然

假设我们现在需要学习整个概率密度，这样以后我们就能回答任何推断问题。
这个情境中我们把学习的目标视為「估计概率密度」。我们想要建构一个 $$p$$，使其
尽可能接近 $$p^* $$。我们怎么测量远近？我们再次动用之前讲变量推断时提到
 KL 分歧，：
{% math %}
KL(p^* || p) = \sum_x p^* (x) \log \frac{p^* (x)}{p(x)} = -H(p^* ) - \mathbb{E}_{x \sim p^* } [ \log p(x) ].
{% endmath %}

其中第一项不依赖 $$p$$，所以要最小化 KL分歧相当于最大化对数似然函数
{% math %}
\mathbb{E}_{x \sim p^* } [ \log p(x) ].
{% endmath %}
这个目标函数要求 $$p$$ 给予那些从 $$p^* $$ 中采样出的值更高的概率，
从而反映真正的分布。
有了这个度量我们可以比较模型，但是因為我们没有计算 $$H(p^* )$$，
我们并不知道一个模型离最优模型有多近。

然而还有一个问题：一般来说 $$p^* $$ 是未知的。所以我们必须用经验指数似然值
（即蒙地卡罗估计）来近似期望的指数似然函数 $$\mathbb{E}_{x \sim p^* } [ \log p(x) ]$$：
{% math %}
\mathbb{E}_{x \sim p^* } [ \log p(x) ] \approx \frac{1}{|D|} \sum_{x \in D}  \log p(x),
{% endmath %}
其中 $$D$$ 是从 $$p^* $$ 中采样的独立同分布(iid)数据集。

最大似然学习法定义如下：
{% math %}
\max_{p \in M} \frac{1}{|D|} \sum_{x \in D}  \log p(x),
{% endmath %}

### 例子

As a simple example, consider estimating the outcome of a biased coin. Our dataset is a set of tosses and our task is to estimate the probability of heads/tails on the next flip. We assume that the process is controlled by a probability distribution $$p^* (x)$$ where $$x \in \{h,t\}$$. Our class of models $$M$$ is going to be the set of all probability distributions over $$\{h,t\}$$.

How should we choose $$p$$ from $$M$$ if 60 out
of 100 tosses are heads? Let's assume that $$p(x=h)=\theta$$ and $$p(x=t)=1−\theta$$. If our observed data is $$D = \{h,h,t,h,t\}$$, our likelihood becomes $$\prod_i p(x_i ) = \theta \cdot \theta \cdot (1 − \theta) \cdot \theta \cdot (1 − \theta)$$; maximizing this yields $$\theta = 60\%$$.

More generally, our log-likelihood function is simply
{% math %}
L(\theta) = \text{# heads} \cdot \log(\theta) + \text{# tails} \cdot \log(1 − \theta),
{% endmath %}
for which the optimal solution is
{% math %}
\theta^* = \frac{\text{# heads}}{\text{# heads} + \text{# tails}}.
{% endmath %}

## 似然、损失与风险函数

我们可以推广这个概念，引进「损失函数」的概念。
损失函数 $$L(x,p)$$ 测量一个模型分布 $$p$$ 在一个数据样本 $$x$$ 上造成的损失。
假设样本是采样 $$p^* $$ 而来，我们的目标是找到最小化预期损失，即「风险」的模型
{% math %}
\mathbb{E}_{x \sim p^* } [ L(x,p) ] \approx \frac{1}{|D|} \sum_{x \in D}  L(x,p),
{% endmath %}
注意对应最大似然估计的损失函数值是对数损失 $$-\log p(x)$$。

Another example of a loss is the conditional log-likelihood. Suppose we want to predict a set of variables $$y$$ given $$x$$, e.g., for segmentation or stereo vision. We concentrate on predicting $$p(y|x)$$, and use a conditional loss function $$L(x,y,p) = −\log p(y \mid  x).$$
Since the loss function only depends on $$p(y \mid  x)$$, it suffices to estimate the conditional distribution, not the joint. This is the objective function we use to train conditional random fields (CRFs).

Suppose next that our ultimate goal is structured prediction, i.e. given
$$x$$ we predict $$y$$ via $$\arg\max_y p(y \mid  x)$$. What loss function should we use to measure error in this setting?

One reasonable choice would be the classification error:
{% math %}
\mathbb{E}_{(x,y)\sim p^*} [\mathbb{I}\{ \exists y' \neq y : p(y'\mid x) \geq p(y\mid x) \}],
{% endmath %}
which is the probability over all $$(x, y)$$ pairs sampled from $$p^*$$ that we predict
the wrong assignment.
A somewhat better choice might be the hamming loss, which counts the number of variables in which the MAP assignment differs from the ground truth label.
There also exists a fascinating line of work on generalizations of the hinge loss to CRFs, which leads to a class of models called *structured support vector machines*.

The moral of the story here is that it often makes sense to choose a loss that is appropriate to the task at hand, e.g. prediction rather than full density estimation.

## Empirical Risk and Overfitting

Empirical risk minimization can easily overfit the data.
The data we have is a sample, and usually there is vast amount of samples that we have never seen. Our model should generalize well to these "never-seen" samples.

### The bias/variance tradeoff

Thus, we typically restrict the *hypothesis space* of distributions that we search over. If the hypothesis space is very limited, it might not be able to represent $$p^*$$, even with unlimited data. This type of limitation is called bias, as the learning is limited on how close it can approximate the target distribution

If we select a highly expressive hypothesis class, we might represent better the data. However, when we have small amount of data, multiple models can fit well, or even better than the true model. Moreover, small perturbations on D will result in very different estimates. This limitation is call the variance.

Thus, there is an inherent bias-variance trade off when selecting the hypothesis class. One of the main challenges of machine learning is to choose a model that is sufficiently rich to be useful, yet not so complex as to overfit the training set.

### How to avoid overfitting?

High bias can be avoided by increasing the capacity of the model. We may avoid high variance using several approaches.

We may impose hard constraints, e.g. by selecting a less expressive hypothesis class: Bayesian networks with at most $$d$$ parents or pairwise (rather than arbitrary-order) MRFs. We may also introduce a soft preference for "simpler" models by adding a regularizer term $$R(p)$$ to the loss $$L(x,p)$$, which will penalize overly complex $$p$$.

### Generalization error

At training, we minimize empirical loss
{% math %}
\frac{1}{|D|} \sum_{x \in D}  \log p(x).
{% endmath %}
However, we are actually interested in minimizing
{% math %}
\mathbb{E}_{x \sim p^*} [ \log p(x) ].
{% endmath %}

We cannot guarantee with certainty the quality of our learned model.
This is because the data $$D$$ is sampled stochastically from $$p^*$$, and we might get an unlucky sample. The goal of learning theory is to prove that the model is approximately correct: for most $$D$$, the learning procedure returns a model whose error is low. There exist a vast literature that quantifies the probability of observing a given error between the empirical and the expected loss given a particular type of model and a particular dataset size.

## Maximum likelihood learning in Bayesian networks

Let us now apply this long discussion to a particular problem of interest: parameter learning in Bayesian networks.

Suppose that we are given a Bayesian network $$p(x) = \prod^n_{i=1} \theta_{x_i \mid pa(x_i)}$$ and i.i.d. samples $$D=\{x^{(1)},x^{(2)},...,x^{(m)}\}$$. What is the maximum likelihood estimate of the parameters (the CPDs)?

We may write the likelihood as
{% math %}
L(\theta, D) = \prod_{i=1}^n \prod_{j=1}^m \theta_{x_i^{(j)} \mid pa(x_i^{(j)})}
{% endmath %}
Taking logs and combining like terms, this becomes
{% math %}
\log L(\theta, D) = \sum_{i=1}^n \#(x_i, pa(x_i)) \cdot \theta_{x_i \mid pa(x_i)}.
{% endmath %}
Thus, maximization of the (log) likelihood function decomposes into separate maximizations for the local conditional distributions!
This is essentially the same as the head/tails example we saw earlier (except with more categories). It's a simple calculus exercise to formally show that
{% math %}
\theta^*_{x_i \mid pa(x_i)} = \frac{\#(x_i, pa(x_i))}{\#(pa(x_i))}.
{% endmath %}

We thus conclude that in Bayesian networks with discrete variables, the maximum-likelihood estimate has a closed-form solution. Even when the variables are not discrete, the task is equally simple: the log-factors are linearly separable, hence the log-likelihood reduces to estimating each of them separately. The simplicity of learning is one of the most convenient features of Bayesian networks.
