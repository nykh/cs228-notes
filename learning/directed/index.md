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
尽可能接近 $$p^* $$。我们怎么测量远近？我们动用之前讲变量推断时提到过的
KL 分歧：
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

另一个损失函数的例子是条件指数似然率。假设我们已知 $$x$$ 想预测一组变量 $$y$$，例如图形分区问题或立体视觉问题。我们集中在预测 $$p(y|x)$$
并使用条件损失函数 $$L(x,y,p) = −\log p(y \mid  x).$$
因為这个条件函数只依赖 $$p(y \mid  x)$$，只需要估计条件分布，
不需要估计整个联合分布。训练条件随机场时使用的是这个目标函数。

假设我们下个目标是结构预测。即已知 $$x$$
我们预测 $$y$$ 取 $$\arg\max_y p(y \mid  x)$$。
这个情境下我们应该用什么样的损失函数？

一个合理的选择是分类错误：
{% math %}
\mathbb{E}_{(x,y)\sim p^* } [\mathbb{I}\{ \exists y' \neq y : p(y'\mid x) \geq p(y\mid x) \}],
{% endmath %}
表示在所有从 $$p^* $$ 中抽样的 $$(x, y)$$ 对当中，模型预测错误赋值的概率。
另一个可能更好的选择是 Hamming 损失，计算的是 MAP 赋值中有几个变量的赋值不同于参考值。
还有一些精彩的工作做的是把铰链损失推广到条件随机场，这类模型称作「结构化支持向量机」。

故事的教训选择一个适合手上任务（预测而不是估计整个密度）的损失函数会比较合理。

## 经验风险与过适

最小化经验风险常常导致模型对数据过适。我们有的数据是一个样本，而通常
我们有很多样本没有见过。我们的模型必须能推广到这些从没见过的样本。

### 偏差/方差的取舍

Thus, we typically restrict the *hypothesis space* of distributions that we search over. If the hypothesis space is very limited, it might not be able to represent $$p^* $$, even with unlimited data. This type of limitation is called bias, as the learning is limited on how close it can approximate the target distribution

If we select a highly expressive hypothesis class, we might represent better the data. However, when we have small amount of data, multiple models can fit well, or even better than the true model. Moreover, small perturbations on D will result in very different estimates. This limitation is call the variance.

Thus, there is an inherent bias-variance trade off when selecting the hypothesis class. One of the main challenges of machine learning is to choose a model that is sufficiently rich to be useful, yet not so complex as to overfit the training set.

### 如何防止过适？

增加模型的变数可以防止偏差过高。而防止高方差有几种作法。

一是可以加上很强的约束，例如选择一个比较没有表现能力的模型类别：
每个节点的父节点不超过 $$d$$ 个的贝叶斯网络、
二元（而不是任意多元）的马可夫网络等等。我们也可以软性地偏好简单的模型，
只要在损失函数 $$L(x,p)$$ 裡加上正规化项 $$R$$，令其惩罚过于复杂的 $$p$$。

### 推广错误

训练期间我们最小化经验损失
{% math %}
\frac{1}{|D|} \sum_{x \in D}  \log p(x).
{% endmath %}
然而，我们真正想最小化的是
{% math %}
\mathbb{E}_{x \sim p^* } [ \log p(x) ].
{% endmath %}

我们不能保证我们学习模型的质量。因為数据 $$D$$ 是从 $$p^* $$ 随机采样出的。
所以可能会得到不好的样本。学习理论的目标就是证明某个模型已经接近正确，对于
大部分 $$D$$，学习过程回传一个错误很低的模型。There exist a vast literature that quantifies the probability of observing a given error between the empirical and the expected loss given a particular type of model and a particular dataset size.

## 贝叶斯网络上的最高似然学习

现在我们把这整篇文章的内容应用在一个有意思的问题上，贝叶斯网络上的参数学习。

已知贝叶斯网络 $$p(x) = \prod^n_{i=1} \theta_{x_i \mid pa(x_i)}$$
及 iid 样本 $$D=\{x^{(1)},x^{(2)},...,x^{(m)}\}$$，
求参数（条件概率表）的最大似然估计。

我们可以把似然写作
{% math %}
L(\theta, D) = \prod_{i=1}^n \prod_{j=1}^m \theta_{x_i^{(j)} \mid pa(x_i^{(j)})}
{% endmath %}
求指数并合并同类项，得
{% math %}
\log L(\theta, D) = \sum_{i=1}^n \#(x_i, pa(x_i)) \cdot \theta_{x_i \mid pa(x_i)}.
{% endmath %}
因此这个(指数)似然函数的最大值可以分解成局部条件分布各自最大值之和！
这与先前的硬币例子是一样的（只是结果有更多种类）。
基本的微积分足以证明：
{% math %}
{\theta^* }_{x_i \mid pa(x_i)} = \frac{\#(x_i, pa(x_i))}{\#(pa(x_i))}.
{% endmath %}

总而言之，对离散变量值的贝叶斯网络，最大似然估计有闭形式解。
就算变量值不是离散任务也一样简单。因為指数因子是线性可分的，所以指数似然函数
可以分解為对因子逐个估计。
容易学习是贝叶斯最方便的特色之一。
