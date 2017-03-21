---
layout: post
title: 使用采样方法
---
实际上，我们常常使用很复杂的概率模型，消除变数法一类简单算法通常太慢。
事实上，很多类有趣的模型都不允许多项式时间的精确解。因此许多研究试图发明算法
来求推断问题的近似解。这个笔记开始讨论这些算法。

有两大类近似算法：「变分方法」{% sidenote 1 '变分方法得名自 *变分法*，研究如何优化其他以函数為定义域的函数。'%}把推断看作一个优化问题。而「采样发法」则通过从一个分布不断产生随机数来近似解。

采样方法可以用来求边缘跟MAP推断，另外还能用来计算许多有趣的量，
例如概率模型中变数的期望值 $$\Exp[f(X)]$$。
传统上采样方法是近似推断的主要方法，然而最近15年间变分方法开始斩露头角，常常还超过采样方法。

## 从概率分布中采样

首先作為暖身，先想想怎么从一个有 $$k$$ 种结果的多项分布中采样，对应的概率是 $$\theta_1,...,\theta_k$$。

一般意义上的采样不是简单的问题。我们的电脑只能从很简单的分布中产生随机数{% sidenote 1 '甚至这些数也不是真正的随机数。通常是自一个确定的数列中产生，只是这个数列某些性质（例如其累计期望值）与真正的随机分布一样。我们把这样的随机数称作偽随机数。'%}，例如实数区间 $$[0,1]$$ 上的均匀分布。
所有的采样技术都利用某些技巧重复调用某些简单的子函数。

In our case, we may reduce sampling from a multinomial variable to sampling a single uniform variable by subdividing a unit interval into $$d$$ regions with region $$i$$ having size $$\theta_i$$. We then sample uniformly from $$[0,1]$$ and return the value of the region in which our sample falls.
{% maincolumn 'assets/img/multinomial-sampling.png' '把采样多项分布简化為采样一个 [0,1] 上的均匀分布'%}

### 从有向概率模型中采样

{% marginfigure 'nb1' 'assets/img/grade-model.png' '描述一个学生在考试中表现的贝叶斯模型。其分布可以表示為各个表格表示的条件概率分布的积。'%}

上述技巧自然适用于有多项分布的贝叶斯网路。我们利用的方法叫做祖先采样或前采样。
对于贝叶斯网络的概率分布 $$p(x_1,...,x_n)$$，我们以拓朴顺序采样各变数。
也就是说我们先对根节点采样，然后在向下一层的条件概率进行采样，并把上一层采样出的值视為已知。以此类推直到所有节点都完成采样。

In our earlier model of a student's grade, we would first sample an exam difficulty $$d'$$ and an intelligence level $$i'$$. Then, once we have samples, $$d', i'$$ we generate a student grade $$g'$$ from $$p(g \mid d', i')$$. At each step, we simply perform standard multinomial sampling.

## 蒙地卡罗估计

Sampling from a distribution lets us perform many useful tasks, including marginal and MAP inference, as well as computing integrals of the form
{% math %}
\Exp_{x \sim p}[f(x)] = \sum_{x} f(x) p(x).
{% endmath %}
If $$g$$ does not have a special structure that matches the Bayes net structure of $$p$$, this integral will be impossible to perform analytically; instead, we will approximate it using a large number of samples from $$p$$. 使用大量取样来求解的算法思想称作「蒙地卡罗」方法{% sidenote 1 'The name Monte Carlo refers to a famous casino in the city of Monaco. The term was originally coined as a codeword by physicists working the atomic bomb as part of the secret Manhattan project.'%}.

蒙地卡罗积分是一般蒙地卡罗方法的一个实例。
这个方法近似一个期望值如下
{% math %}
\Exp_{x \sim p}[f(x)] \approx \frac{1}{T} \sum_{t=1}^T f(x^t),
{% endmath %}
其中 $$x^1,...,x^T$$ 从 $$p$$ 分布中取样。

很容易证明 $$I_T$$ 的期望，即蒙地卡罗期望，与真正的积分相等。我们说 $$I_T$$ 是 $$\Exp_{x \sim p}[f(x)]$$ 的无偏估计。
另外，由于当 $$T \to \infty$$，$$I_T \to \Exp_{x \sim p}[f(x)]$$，我们可以证明 $$I_T$$ 的方差等于 $$\text{var}_P(f(x))/T$$。我们可以通过增加 $$T$$ 使得方差任意小。

### 拒绝采样

{% marginfigure 'nb1' 'assets/img/rejection-sampling.png' 'Graphical illustration of rejection sampling. We may compute the area of circle by drawing uniform samples from the square; the fraction of points that fall in the circle represents its area. This method breaks down if the size of the circle is small relative to the size of the square.'%}
蒙地卡罗积分的特例是拒绝采样。我们可以用它估计区域 $$R$$ 的面积，只要在一个更大已知区域内进行采样然后算有多少样本落入 $$R$$ 当中。

例如我们可以用拒绝采样来计算 $$p(x=x')$$ 形式的边缘概率：我们可以把概率写成 $$\Exp_{x\sim p}[\Ind(x=x')]$$ 然后用蒙地卡罗方法近似。这相当于从 $$p$$ 采样然后只保留那些 $$x$$ 相当于边缘条件的样本。

### 重要性采样

不过这个方法很浪费。例如如果 $$p(x=x')$$ 等于 1%，那么我们就得舍弃掉 99% 的样本。

一个更好的方法是「重要性采样」。主要思想是从分布 $$q$$（大约与 $$f \cdot p$$ 成正比）中采样，然后系统地对样本重新加权，使其总和相当于想要计算的积分。

假设我们想计算 $$\Exp_{x \sim p}[f(x)]$$，我们可以改写积分成
{% math %}
\begin{align*}
\Exp_{x \sim p}[f(x)]
& = \sum_{x} f(x) p(x) \\
& = \sum_{x} f(x) \frac{p(x)}{q(x)} q(x) \\
& = \Exp_{x \sim q}[f(x)w(x)] \\
& \approx \frac{1}{T} \sum_{t=1}^T f(x^t) w(x^t) \\
\end{align*}
{% endmath %}
$$w(x) = \frac{p(x)}{q(x)}$$。换句话说，我们改為从 $$q$$ 取样并以 $$w(t)$$ 重新加权结果，这个蒙地卡罗近似值的期望相等于原本积分的期望。

这一新估计值的方差等于
{% math %}
\text{var}_{x \sim q}(f(x)w(x)) = \Exp_{x \sim q} [f^2(x) w^2(x)] - \Exp_{x \sim q} [f(x) w(x)]^2 \geq 0
{% endmath %}
我们可以把方差归零，只要取 {%m%}q(x) = \frac{|f(x)|p(x)}{\int |f(x)|p(x) dx}{%em%}；这代表只要我们能从 $$q$$ 中采样并计算对应的权重，所有的蒙地卡罗样本都会相等并等于真正积分。当然，从这样的 $$q$$ 采样会是NP困难，但至少给了我们一个方向。

在先前例子中我们要计算 $$p(x=x') = \Exp_{z\sim p}[p(x'|z)]$$，我们可以取 $$q$$ 等于一个均匀分布并使用重要性采样
{% math %}
\begin{align*}
p(x=x')
& = \Exp_{z\sim p}[p(x'|z)] \\
& = \Exp_{z\sim q}[p(x'|z)\frac{p(z)}{q(z)}] \\
& = \Exp_{z\sim q}[\frac{p(x',z)}{q(z)}] \\
& \approx \frac{1}{T} \sum_{t=1}^T \frac{p(z^t, x')}{q(z^t)} \\
\end{align*}
{% endmath %}
不同于拒绝采样，这会使用所有的样本。当 $$p(z|x')$$ 跟均匀分布差不多时，只要几个样本就足够收敛到真正的概率值

## 马可夫链蒙地卡罗法

我们现在把注意力转回如何用采样法进行边缘与MAP推断。我们可以使用一个很强大的技巧，叫做马可夫链蒙地卡罗法o{% sidenote 1 '马可夫链蒙地卡罗法是另一个曼哈顿计画期间发明的算法，几十年后才重新发表。其重要性最近被认為是20世纪[十个最重要算法](https://www.siam.org/pdf/news/637.pdf)之一。'%} (MCMC).

### 马可夫链

马可夫链蒙地卡罗法的关键概念是「马可夫链」。（离散时间）马可夫链是一个随机变量的序列 $$S_0, S_1, S_2, ...$$，其中每个随机变量 $$S_i \in \{1,2,...,d\}$$ 都可以取 $$d$$ 个可能的值。这很直觉的表示了一个系统的状态。初始状态按照 $$P(S_0)$$ 分布。之后每个状态则从前一个状态按照出发按条件概率分布，即 $$S_i$$ 按 $$P(S_i \mid S_{i-1})$$ 分布。

$$P(S_i \mid S_{i-1})$$ 在每一步 $$i$$ 都一样。这表示整个时间过程中转移概率只由现在的状态决定，无关从前的历史。这称作 **马可夫假设**。

{% marginfigure 'mc' 'assets/img/markovchain.png' '三个状态的马可夫链。每条边上的权重表示转移概率。'%}
很容易把转移概率分布表达為一个 $$d \times d$$ 矩阵。
{% math %}
T_{ij} = P(S_\text{new} = i \mid S_\text{prev} = j),
{% endmath %}
其中 $$T^t$$ 表示矩阵幂（多次应用该矩阵运算子 $$t$$）。

如果初始状态 $$S_0$$ 来自矢量概率分布 $$p_0$$，我们可以把 $$t$$ 步后进入每个状态的概率分布 $$p_t$$ 表示為
{% math %}
p_t = T^t p_0.
{% endmath %}

极限 $$\pi = \lim_{t \to \infty} p_t$$ （如果存在）称作马可夫链的「一个」稳定分布。当马可夫链存在稳定分布 $$\pi$$ 并对所有初始状态 $$p_0$$ 都一样我们把这称之為「马可夫链的稳定分布」。

稳定分布存在的一个充分条件是「detailed balance」：
{% math %}
\pi(x') T(x \mid x') = \pi(x) T(x' \mid x) \;\text{for all $x$}
{% endmath %}
容易证明这样的分布 $$\pi$$ 会有均匀分布（只须把等式两边 $$x$$ 边缘化）。然而反过来不成立，因為存在[不满足细致平衡的马可夫链蒙地卡罗法](https://arxiv.org/pdf/1007.2262.pdf).

### Existence of a stationary distribution

The high-level idea of MCMC will be to construct a Markov chain whose states will be joint assignments to the variables in the model and whose stationary distribution will equal the model probability $$p$$.

In order to construct such a chain, we first need to understand when stationary distributions exist. This turns out to be true under two sufficient conditions:

- *Irreducibility*: It is possible to get from any state $$x$$ to any other state $$x'$$ with probability > 0 in a finite number of steps.
- *Aperiodicity*: It is possible to return to any state at any time, i.e. there exists an $$n$$ such that for all $$i$$ and all $$n' \geq n$$, $$P(s_{n'}=i \mid s_0 = i) > 0$$.

The first condition is meant to prevent *absorbing states*, i.e. states from which we can never leave. In the example below, if we start in states $$1,2$$, we will never reach state 4. Conversely, if we start in state 4, then we will never reach states 1,2. If we start the chain in the middle (in state 3), then clearly it cannot have a single limiting distribution.
{% maincolumn 'assets/img/reducible-chain.png' 'A reducible Markov Chain over four states.'%}

The second condition is necessary to rule out transition operators such as
{% math %}
T = \left[
\begin{matrix}
0 & 1 \\
1 & 0
\end{matrix}
\right].
{% endmath %}
Note that this chain alternates forever between states 1 and 2 without ever settling in a stationary distribution.

**Fact**: An irreducible and aperiodic finite-state Markov chain has a stationary distribution.

In the context of continuous variables, the Markov chain must be *ergodic*, which is slightly stronger condition than the above (and which requires irreducibility and aperiodicity). For the sake of generality, we will simply require our Markov Chain to be ergodic.

### Markov Chain Monte Carlo

As we said, the idea of MCMC algorithms is to construct a Markov chain over the assignments to a probability function $$p$$; the chain will have a stationary distribution equal to $$p$$ itself; by running the chain for some number of time, we will thus sample from $$p$$.

At a high level, MCMC algorithms will have the following structure. They take as argument a transition operator $$T$$ specifying a Markov chain whose stationary distribution is $$p$$, and an initial assignment $$x_0$$ to the variables of $$p$$. An MCMC algorithm then perform the following steps.

1. Run the Markov chain from $$x_0$$ for $$B$$ *burn-in* steps.
2. Run the Markov chain from $$x_0$$ for $$N$$ *sampling* steps and collect all the states that it visits.

Assuming $$B$$ is sufficiently large, the latter collection of states will form samples from $$p$$. We may then use these samples for Monte Carlo integration (or in importance sampling). We may also use them to produce Monte Carlo estimates of marginal probabilities. Finally, we may take the sample with the highest probability and use it as an estimate of the mode (i.e. perform MAP inference).


### Metropolis-Hastings 算法

Metropolis-Hastings (MH) 算法是在马可夫链蒙地卡罗当中取样的方法。MH 法的转移操作 $$T(x')$$ 包含两个部份：

- 转移核 $$Q(x'\mid x)$$ 由使用者指定
- 一个接受概率，决定是否接受 $$Q$$ 提案的值，按如下计算
{% math %}
A(x' \mid x) = \min \left(1, \frac{P(x')Q(x \mid x')}{P(x)Q(x' \mid x)} \right).
{% endmath %}

每一步，我们根据 $$Q$$ 选择一个新位置 $$x'$$。然后根据 $$\alpha$$ 决定是否移动到这个新位置或是留在现在位置。

注意接受概率倾向于移动到分布中更加可能的位置（例如想像 $$Q$$ 是均匀分布）。
当 $$Q$$ 提议一个低概率位置的时候，我们就比较不太会移动过去。

实践中，$$Q$$ 通常比较简单，例如集中于当前位置 $$x$$ 的高斯分布。这对连续变量有用。

MH 算法保证对任意 $$Q$$，（目标分布）$$P$$ 会是这个马可夫链的稳定分布。更精确的说，
$$P$$ 对 MH 马可夫链满足细致平衡条件。

要证明，注意当 $$A(x' \mid x) < 1$$，则 $$\frac{P(x)Q(x' \mid x)}{P(x')Q(x \mid x')} > 1$$，于是 $$A(x \mid x') = 1$$。当 $$A(x' \mid x) < 1$$ 有：
{% math %}
\begin{align*}
A(x' \mid x) & =  \frac{P(x')Q(x \mid x')}{P(x)Q(x' \mid x)} \\
P(x')Q(x \mid x') A(x \mid x') & =  P(x)Q(x' \mid x) A(x' \mid x) \\
P(x')T(x \mid x') & =  P(x)T(x' \mid x),
\end{align*}
{% endmath %}
也就是细致平衡方程。我们用 $$T(x \mid x')$$ 表示 MH 的整个转移运算（包含 $$Q$$ 跟 $$A$$）。所以如果 MH 的马可夫链满足遍历性，其稳定分布会成為 $$P$$。

### 吉布斯采样

Metropolis-Hastings 法的一个常见特例是吉布斯采样。对一个变数序列 $$x_1,...,x_n$$ 及初始状态 $$x^0 = (x_1^0,...,x_n^0)$$，我们对变数逐个进行迭代。在每个时间点 $$t$$：

1. 采样 $$x_i' \sim p(x_i \mid x_{-i}^t)$$
2. 设 $$x^{t+1} = (x_1^t, ..., x_i', ..., x_n^t).$$

其中 $$x_{-i}^t$$ 表示 $$x^t$$ 中 $$x_i$$ 以外的所有变数。这步采样通常很容易，因為 $$x_i$$ 只依赖于其马可夫毯，数量通常很小。

吉布斯采样可视為 MH 特例，即使用提议
$$ Q(x_i', x_{-i} \mid x_i, x_{-i}) = P(x_i' \mid x_{-i}). $$。
容易确认在这种情况下接受概率简化至 1（总是接受）。

假设有正确的转移操作，吉布斯采样与MH都会最终得到稳定分布的采样，即 $$P$$。

有些简单的方法保证这点

- 要保证 irreducibility，MH 转移操作 $$Q$$ 应该要能到达每个状态。对吉布斯采样，我们要保证每个 $$x_i'$$ 都能从 $$p(x_i \mid x_{-i}^t)$$ 中采样。
- 保证非周期性，只需要让每个转移有机会待在同样的状态。

实践中不难保证这些条件。

### Running time of MCMC

这个算法中的重要参数是暖身步数 $$B$$。直觉来说，这是系统收敛到极限（稳定）分布所需的步数。这也被称作马可夫链的「混合时间」{% sidenote 1 '这个量有一个详细定义，这裡省略'%}。不巧的是这个时间可以差很多，有时甚至几乎没尽头。例如如果转移矩阵是
{% math %}
T = \left[
\begin{matrix}
1 -\e & \e \\
\e & 1 -\e
\end{matrix}
\right],
{% endmath %}
那么当 $$\e$$ 很小时系统会花很久才能达到稳定分布 $$(0.5, 0.5)$$。每一步上系统都有巨大的可能性待在同样的状态，所以每一个状态都会花很多时间。虽然最终这些状态会收敛到 $$(0.5, 0.5)$$，但这会花很长时间。

This problem will also occur with complicated distributions that have two distinct and narrow modes; with high probability, the algorithm will sample from a given mode for a very long time. These examples are indications that sampling is a hard problem in general, and MCMC does not give us a free lunch. Nonetheless, for many real-world distributions, sampling will produce very useful solutions.

一个可能更重要的问题时我们不知道暖身应该花多少时间，即便理论告诉我们这不用很长。
有很多啟发性的方法估计一个马可夫链是否已经充分混合了。然而，这些啟发性的方法通常需要印出来然后让人来决定，量化测量法都不见得比这个方法可靠。

总之，虽然马可夫链蒙地卡罗法可以让我们从各种分布中采样（从而解决推断问题），有时候运行时间很长，而且没有明确的方法估计要多少步计算才能得到好的结果。
