---
layout: post
title: 最大後驗概率推断
---
这篇笔记会更加深入讨论MAP(最大后验概率)推断问题。
图模型中的最大后验概率推断可以看作以下优化问题：
{% math %}
\max_x \log p(x) = \max_x \sum_c \phi_c(\bfx_c) - \log Z.
{% endmath %}

上一篇笔记中，我们简短介绍了怎么跟计算边缘推断一样使用信息传递的框架来解决MAP推断问题。这篇将会讲一些更高效的专门算法。

## MAP推断的挑战性

某种意义上来说，MAP推断比边缘推断简单。一个原因是因為标准化常数$$\log Z$$ 与$$x$$无关，可以不用计算。
{% math %}
\arg \max_x \sum_c \phi_c(\bfx_c).
{% endmath %}

边缘推断可以被视為计算一个模型所有赋值的概率然后求和。这些赋值中有一个对应最大后验概率，所以如果我们把求和运算替换為求最大值运算，就能求得概率最高的赋值。然而这种基于穷举的算法不是最有效率的。

值得注意的是，MAP推断仍然不是一般意义上的简单问题。上述的优化问题的特例中包含了许多难解的问题，例如**3SAT**问题。我们可以将一个3SAT问题简化成MAP推断问题：对于3SAT的每个条件式 $$c = (x \lor y \lor \neg z)$$ 建构一个因子 $$\theta_c (x, y, z)$$，当 $$x, y, z$$ 满足条件 $$c$$ 时取1，否则取0。如此一来，这个3SAT问题有解当且仅当对应MAP的赋值相等于条件的数目{% sidenote 1 '我们也可以类似地证明边缘推断是NP困难的。主要思想是加入一个变量$$X$$，当所有条件皆满足时取1，否则取0。则边缘概率大于0当且仅当对应3SAT问题有解。'%}。

不论如何，可以证明MAP问题比一般的推断问题简单，因為有些模型的MAP推断问题是多项式时间内可解而一般推断问题是NP困难的。

### 例子

MAP许多有趣的例子来自*结构预测*。要解决结构预测问题需要在一个CRF模型$$p(y|x)$$上推断：
{% math %}
\arg \max_y \log p(y|x) =  \arg \max_y \sum_c \phi_c(\bfy_c, \bfx_c).
{% endmath %}

{% marginfigure 'ocr' 'assets/img/ocr.png' 'OCR用的单链型CRF' %}
之前讲到CRF时我们已经深入讨论了结构预测问题。记得我们主要的例子是识别手写字：输入英文字母的图像矩阵 $$x_i \in [0, 1]^{d\times d}$$，这个框架下的MAP推断相等于联合地识别图像中最有可能的词 $$(y_i)_{i=1}^n$$

另一个例子是图形分区。在分区问题裡，我们想要识别图像中的物体并标记所有属于该物体的像素。输入 $$\bfx \in [0, 1]^{d\times d}$$ 是像素矩阵，我们的任务是预测标记矩阵 $$y \in \{0, 1\}^{d\times d}$$，1表示$$x$$的对应像素属于物体。直觉来说，邻近的像素应该有相似的标记（一匹马的像素应该是连续体而不是到处都有的白噪音）。
{% marginfigure 'ocr' 'assets/img/imagesegmentation.png' '图形分区问题示例' %}

我们可以很自然地用使用[Potts 模型](https://en.wikipedia.org/wiki/Potts_model)表达这个先验知识。在之前的例子中，我们可以引入势 $$\phi(y_i,x)$$ 表达一个像素属于物体的似然率。然后我们用成对的势 $$\phi(y_i, y_j )$$ 来表达邻近像素的标记 $$y_i, y_j$$，以强调相邻的$$y$$有更高概率具有相同的标记值。

## 图割

现在我们讨论一个适用特定Potts模型的，高效的精确MAP推断算法。与之前的方法（例如联合树）不同，就算模型有很高的树宽度，这个算法仍然可以接受。

假设我们有二进制二元马可夫网络，其中边能量（边因子的对数）有以下形式
{% math %}
E_{uv}(x_u, x_v) =
\begin{cases}
0 & \; \text{if} \; x_u = x_v \\
\lambda_{st} & \; \text{if} \; x_u \neq x_v,
\end{cases}
{% endmath %}
其中成本 $$\lambda_{st} \geq 0$$ 惩罚边不匹配的边势。
再假设每个节点有一元势。我们可以标准化节点的能量，以使 $$E_u(1) = 0$$ 或者 $$E_u(0) = 0$$ 并且 $$E_u \geq 0$$。


{% marginfigure 'mincut' 'assets/img/mincut.png' '一个2x2 马可夫网络上的图像分区任务可以表示為一个图割问题。虚线边属于最小割。（来源: Machine Learning: A Probabilistic Perspective）' %}
这个模型的发明要感谢图形分区问题。我们想求一个赋值使得能量最小，这也会试图减少相邻变量间的不协调。

可以把这类模型的能量最小化过程看作是在一个增补图$$G'$$中的最小割问题：
在概率图模型中加入源与汇节点$$s,t$$；令$$s$$连结到 $$E_u(0) = 0$$ 的节点$$u$$，其边权重為 $$E_u(1)$$；又令 $$E_v(1) = 0$$ 的节点$$v$$连结到$$t$$，其边权重為 $$E_v(0)$$。最后将所有原图中的边的权重设為 $$E_{uv}$$。

很容易理解这个建构使得图中最小割的费用等于原本模型的最小能量。特别的是，切割后所有在源一侧的节点都取$$0$$，而汇一侧的节点都取$$1$$。所有连接两个不同值节点的边组成最小割的集合。

在一个图$$G=(V,E)$$上计算最小割最快算法需要$$O(EV\log V)$$或$$O(V^3)$$。类似的技巧也适用于更加一般的模型，只要它们的边势满足「分模」（*submodular*）性质。详情请见教科书(例如Koller and Friedman)。

## 基于线性规划的方法

虽然图分割方法可以求得精确的MAP赋值，却只适用特定种类的马可夫网络。我们接下来介绍一个适用范围更广的近似算法。

### 线性规划

我们的第一个近似推断策略是把MAP推断归约成整数线性规划。**线性规划**指的是下面类别的问题
{% math %}
\begin{align*}
\min \;& \bf{c} \cdot \bfx \\
\textrm{s.t. } & A \bfx \leq \bf{b}
\end{align*}
{% endmath %}
其中$$\bfx \in \mathbb{R}^n$$、$$\bf{c}, \bf{b} \in \mathbb{R}^n$$、$$A\in \mathbb{R}^{n\times n}$$是参数。

这类问题在理工各领域都很常见，在30年代已经研究得很透彻，产生了一个很庞大的理论{% sidenote 1 "80年代应用数学的一个重大突破是发现了线性规划的多项式时间[算法](https://en.wikipedia.org/wiki/Karmarkar%27s_algorithm)。"%}及实用[工具](https://en.wikipedia.org/wiki/CPLEX)，用来在可接受的时间内解很大(十万或更多变数)的线性规划问题实例。

整数线性规划(ILP)是线性规划问题加上条件 $$\bfx \in \{0, 1\}^n$$。不妙的是这个附加条件使得优化过程明显变难，一般情况下变成NP困难。然而实际上有许多啟发技巧可以对ILP问题求解。有些商用求解程序可以处理几千个变量的实例。

整数线性规划求解问题的主要技巧是*舍入*。主要思想是把 $$\bfx \in \{0, 1\}^n$$ 这个条件放松成 $$0 \leq \bfx \leq 1$$ 求解，再把解四舍五入成整数值。这个方法在实践中效果出奇得好，并且对某些种类的整数线性规划问题有理论保证。

### 把MAP推断问题表达為整数线性规划

為求简单我们考虑一个在二元马可夫网络中的MAP问题。我们可以把MAP目标简化成整数线性规划，只须引入两种指示函数。

- 对每个节点 $$i \in V$$ 的状态 $$x_i$$ 引入变数 $$\mu_i(x_i)$$
- 对每条边 $$(i,j) \in E$$ 对应的状态$$x_i,x_j$$引入变数 $$\mu_{ij}(x_i,x_j)$$

有了这些变数我们就能把MAP目标改写為
{% math %}
\max_\mu \sum_{i \in V} \sum_{x_i} \theta_i (x_i) \mu_i(x_i) + \sum_{i,j \in E} \sum_{x_i, x_j} \theta_{ij} (x_i, x_j) \mu_{ij}(x_i, x_j).
{% endmath %}

為了在这些 $$\mu$$ 的基础上优化我们还必须引入条件。
首先，我们强迫每个集群取一个本地赋值：
{% math %}
\begin{align*}
\mu_i(x_i) & \in \{0,1\} \; \forall \, i, x_i \\
\sum_{x_i} \mu_i(x_i) & = 1\; \forall \, i \\
\mu_{ij}(x_i,x_j) & \in \{0,1\} \; \forall \, i,j \in E, x_i, x_j \\
\sum_{x_i, x_j} \mu_{ij}(x_i, x_j) & = 1 \; \forall \, i,j \in E.
\end{align*}
{% endmath %}

这些赋值还必须有一致性：
{% math %}
\begin{align*}
\sum_{x_i} \mu_{ij}(x_i, x_j) & = \mu_j(x_j) \; \forall \, i,j \in E, x_j \\
\sum_{x_j} \mu_{ij}(x_i, x_j) & = \mu_i(x_i) \; \forall \, i,j \in E, x_i
\end{align*}
{% endmath %}

这些约束与MAP目标函数一同定义了一个整数线性规划问题，其解相等于MAP赋值。这个整数线性规划问题实例仍然是NP困难的，但我们有办法把它放松成（简单可解的）线性规划问题。这就是基于线性规划的MAP方法的主要思想

一般而言，这个方法仅能求近似解。重要的例外是对树型图而言，放松可以保证得出整数解，因此这个解是最优的。{% sidenote 1 "例如见课本 Koller and Friedman，内有证明与深入讨论。"%}.

## 对偶分解方法

现在我们看怎么把MAP目标转化成更适合的优化问题。
假设有如下形式的马可夫网络
{% math %}
\max_x \sum_{i \in V} \theta_i (x_i) + \sum_{f \in F} \theta_{f}(x_f),
{% endmath %}
其中 $$F$$ 表示某个因子（例如二元马可夫网络中的边势）{% sidenote 1 "这些简短的推导大致基于[Sontag等人](http://cs.nyu.edu/~dsontag/papers/SonGloJaa_optbook.pdf)的教学，有兴趣看完整内容的读者可以参阅。"%}。我们让 $$p^*$$ 表示这个目标函数的最优值，$$x^*$$ 表示最优赋值。

上述的目标很难优化，因為势与势之间存在耦合。现在我们先考虑另一个目标函数，单独最优化每个势：
{% math %}
\sum_{i \in V} \max_{x_i}  \theta_i (x_i) + \sum_{f \in F} \max_{x^f} \theta_{f} (x^f) .
{% endmath %}

优化这个很简单，但是结果只是真正MAP赋值的一个上界。要让我们的松弛更，我们需要引入势之间一致的约束：
{% math %}
x^f_i - x_i = 0 \; \forall f, \forall i \in f.
{% endmath %}
对偶分解方法主要思想是放松这些约束，从而在这两个目标函数之间取得平衡。

首先我们取含约束条件问题的**拉格朗日乘数**，即
{% math %}
L(\delta, \bfx^f, \bfx) = \sum_{i \in V}  \theta_i (x_i) + \sum_{f \in F} \theta_{f} (x^f) + \sum_{f \in F} \sum_{i \in f} \sum_{x'_i} \delta_{fi}(x_i')\left( \Ind_{x'_i = x_i} - \Ind_{x'_i = x^f_i} \right).
{% endmath %}

变数 $$\delta$$ 称作拉格朗日乘数。每一个 $$\delta$$ 都对应一个条件{% sidenote 1 "围绕拉格朗日乘数有一个很深刻且强大的理论，用来在约束条件下进行优化。更加深入的内容参见[凸优化课程](http://stanford.edu/class/ee364a/)"%}。
注意 $$x, x^f = x^*$$ 是拉格朗日乘数的一个合法赋值。其对应目标函数值相当于任意 $$\delta$$ 的 $$p^*$$，
因為拉格朗日乘数只是乘以 0。可见拉格朗日是 $$p^*$$ 的上界：
{% math %}
L(\delta) := \max_{\bfx^f, \bfx} L(\delta, \bfx^f, \bfx) \geq p^* \; \forall \delta.
{% endmath %}

為了求得上确界，我们在 $$\delta$$ 上优化 $$L(\delta)$$。根据拉格朗日对偶性理论，最佳的 $$\delta^*$$ 对应上确界，即
{% math %}
L(\delta^*) =  p^*.
{% endmath %}

在我们的问题中这其实不难证明。注意我们可以把拉格朗日乘数重新设定為：
{% math %}
\begin{align*}
L(\delta)
& = \sum_{i \in V} \max_{x_i} \left( \theta_i (x_i) + \sum_{f:i \in f} \delta_{fi}(x_i) \right) + \sum_{f \in F} \max_{x^f} \left( \theta_f (x^f) + \sum_{i \in f} \delta_{fi}(x_i) \right) \\
& := \sum_{i \in V} \max_{x_i} \bar \theta_{i}^\delta (x_i) + \sum_{f \in F} \max_{x^f} \bar \theta_{f}^\delta (x^f).
\end{align*}
{% endmath %}

假设我们能找到对偶变量 $$\bar \delta$$，使得局部最大值对应的参数 $$\bar \theta_{i}^{\bar \delta} (x_i)$$ 和 $$\bar \theta_{f}^{\bar \delta} (x^f)$$ 一致，也就是说，假设我们能找到 $$\bar x$$，使得 $$\bar x_i \in \arg\max_{x_i} \bar \theta_{i}^{\bar \delta} (x_i)$$ 且 $$\bar x^f \in \arg\max_{x^f} \bar \theta_{f}^{\bar \delta} (x^f)$$。则有
{% math %}
L(\bar \delta) =  \sum_{i \in V} \bar \theta_{i}^{\bar\delta} (\bar x_i) + \sum_{f \in F} \bar \theta_{f}^{\bar\delta} (\bar x^f) =  \sum_{i \in V} \theta_{i} (\bar x_i) + \sum_{f \in F} \theta_{f} (\bar x^f).
{% endmath %}
第一个等式来自 $$L(\delta)$$ 的定义，第二个是因為拉格朗日乘数项在$$x$$和 $$x^f$$一致时相消

另一方面，由 $$p^*$$ 的定义有
{% math %}
\sum_{i \in V} \theta_{i} (\bar x_i) + \sum_{f \in F} \theta_{f} (\bar x^f) \leq p^* \leq L(\bar\delta)
{% endmath %}
蕴含 $$L(\bar\delta) = p^*$$.

以上论证说明两件事：

- 拉格朗日乘数求得的上界只要对应合适的$$\delta$$就可以是上确界
- 要计算 $$p^*$$ 只需要求 $$\delta$$ 使得局部子问题彼此一致，实践中这很容易达到。

### 最小化目标函数

这裡简介几种计算 $$L(\delta^*)$$ 的方法。

因為目标函数 $$L(\delta)$$ 是连续凹函数{% sidenote 1 "这个目标函数对每个点的仿射函数取最大值。"%}，我们可以用子梯度下降法来最小化。
令 $$\bar x_i \in \arg\max_{x_i} \bar \theta_{i}^{\delta} (x_i)$$ 和 $$\bar x^f \in \arg\max_{x^f} \bar \theta_{f}^\delta (x^f)$$。
可以证明 $$L(\delta)$$ 关于 $$\delta_{fi}(x_i)$$ 的梯度 $$g_{fi}(x_i)$$  等于 $$1$$ 如果 $$\bar x_i^f \neq \bar x_i$$，否则等于 0。
类似的 $$g_{fi}(x_i^f)$$ 等于 $$-1$$ 若 $$\bar x_i^f \neq \bar x_i$$ 否则等于 0。
这个表达式会降低 $$\bar \theta_{i}^{\delta} (\bar x_i)$$ 并增大 $$\bar \theta_{f}^\delta (\bar x^f)$$，直到他们彼此接近。

要计算这些梯度我们进行运算 $$\bar x_i \in \arg\max_{x_i} \bar \theta_{i}^{\delta} (x_i)$$ 以及 $$\bar x^f \in \arg\max_{x^f} \bar \theta_{f}^\delta (x^f)$$。
有一些有用的特殊情况下这是可能的，例如当因数的作用域很小、图的树宽度很窄、或是当因数在大部份定义域上不变时。

另一个最小化 $$L(\delta)$$ 的方法是使用区块坐标下降（block coordinate descent）。典型产生区块的方法是考虑某个特定因子 $$f$$ 对应的所有变数 $$\delta_{fi}(x_i)$$。这样得出的迭代很类似有环的最大值-积信念传播。实践中这个方法可能比子梯度下降法更快，并且保证每次迭代目标函数都会下降，并且不需要速度参数。区块坐标下降的缺点是不一定会找到全局最小值（因為目标函数不是「*强*」凹函数）

### 恢复MAP赋值

如上所述，只要一个解  $$\bfx, \bfx^f$$ 的因子对于某个 $$\delta$$ 相等，我们就能保证这个解是最优的。

如果最优的 $$\bfx, \bfx^f$$ 不相等，要从这对解中得到 MAP 赋值仍然是个 NP完全问题。
但是实践中并不是那么困难。从理论保证上来说，如果每个 $$\bar \theta_i^{\delta^*}$$ 有独特的最大值，那么这个问题是可决定的。
如果不是所有变数都能保证如此，我们可以抓住那些可以解析到最优值的变量，再用精确推断得到剩下的变量。

## 其他方法

### 局部搜索

一个更加啟发式的方法，从一个随机赋值出发然后在赋值的空间裡往局部增加概率的方向「移动」。这个方法没有理论保证，但是依靠先验知识我们通常可以找出很有效的移动方式。所以实践中局部搜索可以很有效。

### Branch and bound

或者也可以穷举所有赋值，并一边淘汰掉显然不含MAP赋值的分支。
可以使用线性规划松弛或是其对偶问题得到上界来简枝。

### 模拟退火

第三个作法是用取样方法（例如Metropolis-Hastings）来从 $$p_t(x) \propto \exp(\frac{1}{t} \sum_{c \in C} \theta_c (x_c ))$$ 中取样。
其中参数 $$t$$ 称作「温度」。当 $$t \to \infty $$，$$p_t$$ 会接近均匀分布，很容易取样。当 $$t \to 0$$，$$p_t$$ 会更加偏向 $$\arg\max_\bfx \sum_{c \in C} \theta_c (x_c )$$，即我们想求得的值。然而因為后者有明显的峰值，这会很难取样。

模拟退火的基本思想是运行一个采样算法，从很高的 $$t$$ 出发，然后慢慢减少 $$t$$ 。
如果「冷却」足够慢，就有保证会找到分布的眾数。然而实践中找到冷却速度需要很多微调，因此不是很容易使用。
