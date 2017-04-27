---
layout: post
title: 贝叶斯网络
---
我们从「表现」这个主题开始。表现讨论的是如何选取一个概率分布对现实世界建模。
想出一个好的模型有时很困难：我们在导论中见过一个分类垃圾邮件的朴素模型，
那个模型需要我们选取的参数数量随英语单词数几何增长！

本章我们讨论防止这种麻烦的一种方法。我们将：

- 学习一个有效且通用的方法，只用几个参数来决定一个概率分布
- 优雅地使用有向无环图来表示这些模型
- 学习有向无环图结构与建模假设之间的关系。这不仅会使这些建模假设更明显，也会
帮助我们设计更加有效的推断算法。

我们这裡讨论的一类模型称作「贝叶斯网络」。下一章节我们会讨论第二种模型，称作「马可夫网络」，使用的是无向图。

## 用贝叶斯网络进行概率建模

有向无环图模型即贝叶斯网络是一类概率模型，可以用很少的参数描述，并表示為
一个有向图。

这种参数化的思想出奇简单。

记得链式法则，我们可以把概率 $$p$$ 写作
{% math %}
p(x_1,x_2,...,x_n) = p(x_1) p(x_2\mid x_1) \cdots p(x_n\mid x_{n-1},...,x_2,x_1).
{% endmath %}
一个紧凑的贝叶斯网络是一种分布，其右手边的因子都只关于几个「祖先变量」 $$x_{A_i}$$:
{% math %}
p(x_i \mid  x_{i-1}...x_1) = p(x_i \mid  x_{A_i}).
{% endmath %}
例如，在一个有五个变量的模型，我们选择表示因子 {%m%}p(x_5\mid x_4, x_3, x_2, x_1){%em%} 為 {%m%}p(x_5 \mid  x_4, x_3){%em%}。在这个例子中，
可以把其祖先写作 $$x_{A_i} = \{x_4, x_3\}$$.

对离散变量（大部分问题都是如此），可以把因子 {%m%}p(x_i\mid x_{A_i}){%em%} 表示為一个「概率表」，每一列{%sidenote 1 '译注：按中国大陆横為「行」直為「列」。台湾相反，见[此处讨论](https://ccjou.wordpress.com/2012/04/17/%E5%85%A9%E5%B2%B8%E7%B7%9A%E6%80%A7%E4%BB%A3%E6%95%B8%E7%9A%84%E7%BF%BB%E8%AD%AF%E5%90%8D%E8%A9%9E%E5%8F%83%E7%85%A7/)。' %}对应 $$x_{A_i}$$ 的赋值，
行对应 $$x_i$$ 的赋值，而每一个表格元素是实际的概率 {%m%}p(x_i\mid x_{A_i}){%em%}。
如果每个变量可以取 $$d$$ 个不同值并有最多 $$k$$ 个祖先变量，则
整个表格会包含最多 $$O(d^{k+1})$$ 个概率。
因為每个变量需要一张表格，整个概率分布可以用 $$O(nd^k)$$ 个参数紧凑表示
（对比朴素作法的 $$O(d^n)$$ 个参数）

### 图表示

这样的分布可以自然表示為有向图，其中每个节点表示变量 $$x_i$$，
边表示依赖关系。特别地，每个节点 $$x_i$$ 的父节点对应其祖先变量 $$x_{A_i}$$。

举例，consider a model of a student's grade $$g$$ on an exam; this grade depends on several factors: the exam's difficulty $$d$$, the student's intelligence $$i$$, his SAT score $$s$$; it also affects the quality $$l$$ of the reference letter from the professor who taught the course. Each variable is binary, except for $$g$$, which takes 3 possible values.{% marginfigure 'nb1' 'assets/img/grade-model.png' 'Bayes net model describing the performance of a student on an exam. The distribution can be represented a product of conditional probability distributions specified by tables. The form of these distributions is described by edges in the graph.'%} The joint probability distribution over the 5 variables naturally factorizes as follows:
{% math %}
p(l, g, i, d, s) = p(l \mid  g) p(g \mid  i, d) p(i) p(d) p(s\mid d).
{% endmath %}
The graphical representation of this distribution is a DAG that visually specifies how random variables depend on each other. The graph clearly indicates that the letter depends on the grade, which in turn depends on the student's intelligence and the difficulty of the exam.

Another way to interpret directed graphs is in terms of stories for how the data was generated.
In the above example, to determine the quality of the reference letter, we may first sample an intelligence level and an exam difficulty; then, a student's grade is sampled given these parameters; finally, the recommendation letter is generated based on that grade.

In the previous spam classification example, we implicitly postulated that email is generated according to a two-step process:
first, we choose a spam/non-spam label $$y$$; then we sample independently whether each word is present, conditioned on that label.

### 严谨定义

更严谨的，贝叶斯网络是一个有向图 $$G = (V,E)$$ 加上

- 每个节点 $$i \in V$$ 对应的一个随机变量
- 每个节点对应一个条件概率分布（CPD） {%m%}p(x_i \mid  x_{A_i}){%em%}，定义 $$x_i$$ 已知其父节点取值时的概率分布。

因此，一个贝叶斯网络定义了一个概率分布 $$p$$。
相反地我们也可以说一个概率分布 $$p$$ 在有向无环图 $$G$$ 上**分解**，
若该概率可以按照图 $$G$$ 分解成一些因子的积。

It is not hard to see that a probability represented by a Bayesian network will be valid: clearly, it will be non-negative and one can show using an induction argument (and using the fact that the CPDs are valid probabilities) that the sum over all variable assignments will be one.
Conversely, we can also show by counter-example that when $$G$$ contains cycles, its associated probability may not sum to one.

## The dependencies of a Bayes net

To summarize, Bayesian networks represent probability distributions that can be formed via products of smaller, local conditional probability distributions (one for each variable).
By expressing a probability in this form, we are introducing into our model assumptions that certain variables are independent.

This raises the question: which independence assumptions are we exactly making by using a model Bayesian network with a given structure described by $$G$$?
This question is important for two reasons: we should know precisely what model assumptions we are making (and whether they are correct); also, this information will help us design more efficient inference algorithms later on.

Let us use the notation $$I(p)$$ to denote the set of all conditional independencies that hold for a joint distribution $$p$$. For example, if $$p(x,y)=p(x)p(y)$$, then we say that $$x \perp y \in I(p)$$.

### Independencies described by directed graphs

It turns our that a Bayesian network $$p$$ very elegantly describes many independencies in $$I(p)$$; these independencies can be recovered from the graph by looking at three types of structures.

For simplicity, let's start by looking at a Bayes net $$G$$ with three nodes: $$A$$, $$B$$, and $$C$$. In this case, $$G$$ essentially has only three possible structures, each of which leads to different independence assumptions. The interested reader can easily prove these results using a bit of algebra.

- {% marginfigure 'bn' 'assets/img/3node-bayesnets.png' 'Bayesian networks over three variables, encoding different types of dependencies: cascade (a,b), common parent (c), and v-structure (d).' %}*Common parent.* If $$G$$ is of the form $$A \leftarrow B \rightarrow C$$, and $$B$$ is observed, then {%m%}A \perp C \mid  B{%em%}. However, if $$B$$ is unobserved, then $$A \not\perp C$$. Intuitively this stems from the fact that $$B$$ contains all the information that determines the outcomes of $$A$$ and $$C$$; once it is observed, there is nothing else that affects these variables' outcomes.
- *Cascade*: If $$G$$ equals $$A \rightarrow B \rightarrow C$$, and $$B$$ is again observed, then, again {%m%}A \perp C \mid  B{%em%}. However, if $$B$$ is unobserved, then $$A \not\perp C$$. Here, the intuition is again that $$B$$ holds all the information that determines the outcome of $$C$$; thus, it does not matter what value $$A$$ takes.
- *V-structure* (also known as *explaining away*): If $$G$$ is $$A \rightarrow C \leftarrow B$$, then knowing $$C$$ couples $$A$$ and $$B$$. In other words, $$A \perp B$$ if $$C$$ is unobserved, but {%m%}A \not\perp B \mid  C{%em%} if $$C$$ is observed.

The latter case requires additional explanation. Suppose that $$C$$ is a Boolean variable that indicates whether our lawn is wet one morning; $$A$$ and $$B$$ are two explanation for it being wet: either it rained (indicated by $$A$$), or the sprinkler turned on (indicated by $$B$$). If we know that the grass is wet ($$C$$ is true) and the sprinkler didn't go on ($$B$$ is false), then the probability that $$A$$ is true must be one, because that is the only other possible explanation. Hence, $$C$$ and $$A$$ are not independent given $$B$$.

These structures clearly describe the independencies encoded by a three-variable Bayesian net. We can extend them to general networks by applying them recursively over any larger graph. This leads to a notion called $$d$$-separation (where $$d$$ stands for directed).

We say that $$Q$$, $$W$$ are $$d$$-separated when variables $$O$$ are observed if they are not connected by an *active path*. An undirected path in the Bayesian Network structure $$G$$ is called *active* given observed variables $$O$$ if for every consecutive triple of variables $$X,Y,Z$$ on the path, one of the following holds:

- $$X \leftarrow Y \leftarrow Z$$, and $$Y$$ is unobserved $$Y \not\in O$$
- $$X \rightarrow Y \rightarrow Z$$, and $$Y$$ is unobserved $$Y \not\in O$$
- $$X \leftarrow Y \rightarrow Z$$, and $$Y$$ is unobserved $$Y \not\in O$$
- $$X \rightarrow Y \leftarrow Z$$, and $$Y$$ or any of its descendents are observed.
{% marginfigure 'dp2' 'assets/img/dsep2.png' 'In this example, $$X_1$$ and $$X_6$$ are $$d$$-separated given $$X_2, X_3$$.' %}{% marginfigure 'dp1' 'assets/img/dsep1.png' 'However, $$X_2, X_3$$ are not $$d$$-separated given $$X_1, X_6$$. There is an active pass which passed through the V-structure created when $$X_6$$ is observed.' %}

For example, in the graph below, $$X_1$$ and $$X_6$$ are $$d$$-separated given $$X_2, X_3$$. However, $$X_2, X_3$$ are not $$d$$-separated given $$X_1, X_6$$, because we can find an active path $$(X_2, X_6, X_5, X_3)$$


The notion of $$d$$-separation is  useful, because it lets us describe a large fraction of the dependencies that hold in our model.
Let {%m%}I(G) = \{(X \perp Y \mid  Z) : \text{$$X,Y$$ are $$d$$-sep given $$Z$$}\}{%em%} be a set of variables that are $$d$$-separated in $$G$$.

**Fact**{% sidenote 2 'We will not formally prove this, but the intuition is that if $$X,Y$$ and $$Y,Z$$ are mutually dependent, so are $$X,Z$$. Thus we can look at adjacent nodes and propagate dependencies according to the local dependency structures outlined above.'%}:
If $$p$$ factorizes over $$G$$, then $$I(G) \subseteq I(p)$$. In this case, we say that $$G$$ is an $$I$$-map (independence map) for $$p$$

In other words, all the independencies encoded in $$G$$ are sound: variables that are $$d$$-separated in $$G$$ are truly independent in $$p$$. However, the converse is not true: a distribution may factorize over $$G$$, yet have independencies that are not captured in $$G$$.

In a way this is almost a trivial statement. If $$p(x,y) = p(x)p(y)$$, then this distribution still factorizes over the graph $$y \rightarrow x$$, since we can always write it as {%m%}p(x,y) = p(x\mid y)p(y){%em%} with a CPD {%m%}p(x\mid y){%em%} in which the probability of $$x$$ does not actually vary with $$y$$. However, we can construct a graph that matches the structure of $$p$$ by simply removing that unnecessary edge.

### The representational power of directed graphs

This raises our last and perhaps most important question: can directed graphs express all the independencies of any distribution $$p$$? More formally, given a distribution $$p$$, can we construct a graph $$G$$ such that $$I(G) = I(p)$$?

First, note that it is very easy to construct a $$G$$ such that $$I(G) \subseteq I(p)$$. A fully connected DAG $$G$$ is an I-map for any distribution since $$I(G) = \emptyset$$. {% marginfigure 'dp1' 'assets/img/full-bayesnet.png' 'A fully connected Bayesian network over four variables. There are no independencies in this model, and it is an I-map for any distribution.' %}

A more interesting question is can we find a *minimal* $$I$$-map $$G$$ for $$p$$, i.e. an $$I$$-map $$G$$ such that the removal of even a single edge from $$G$$ will result in it no longer being an $$I$$-map. This is quite easy: we may start with a fully connected $$G$$ and remove edges until $$G$$ is no longer an $$I$$-map. One way to do this is by following the natural topological ordering of the graph, and removing node ancestors until this is no longer possible; we will revisit this pruning method towards the end of course when performing structure learning.

However, what we are truly interested in is to determine whether the probability $$p$$ admits a *perfect* map $$G$$ for which $$I(p)=I(G)$$. Unfortunately, the answer is no. For example, consider the following distribution $$p$$ over three variables $$X,Y,Z$$: we sample $$X,Y \sim \text{Ber}(0.5)$$ from a Bernoulli distribution, and we set $$Z = X \;\text{ xor }\; Y$$ (we call this the noisy-xor example). One can check using some algebra $$\{X \perp Y, Z \perp Y, X \perp Z\} \in I(p)$$ but $$Z \perp \{Y,X\} \not\in I(p)$$. Thus, $$ X \rightarrow Z \leftarrow Y $$ is an I-map for $$p$$, but none of the 3-node graph structures that we discussed perfectly describes $$I(p)$$, and hence this distribution doesn't have a perfect map.

A related question is whether perfect maps are unique when they exist. Again, this is not the case, as $$X \rightarrow Y$$ and $$X \leftarrow Y$$ encode the same independencies, yet form different graphs. More generally, we say that two Bayes nets $$G_1, G_2$$ are I-equivalent if they encode the same dependencies $$I(G_1) = I(G_2)$$.

When are two Bayesian nets I-equivalent? To answer this, let's return to a simple example with three variables. We say that each of the graphs below have the same *skeleton*, meaning that if we drop the directionality of the arrows, we obtain the same undirected graph in each case.
{% maincolumn 'assets/img/3node-bayesnets.png' 'Bayesian networks over three variables' %}
The cascade-type structures (a,b) are clearly symmetric and the directionality of arrows does not matter. In fact, (a,b,c) encode exactly the same dependencies. We can change the directions of the arrows as long as we don't turn them into a V-structure (d). When we do have a V-structure, however, we cannot change any arrows: structure (d) is the only one that describes the dependency $$X \not\perp Y \mid Z$$. These examples provide intuitions of the following general results on I-equivalence.

**Fact:**
If $$G,G'$$ have the same skeleton and the same v-structures, then $$I(G) = I(G').$$

Again, it is easy to understand intuitively why this is true. Two graphs are I-equivalent if the $$d$$-separation between variables is the same. We can flip the directionality of any edge, unless it forms a v-structure, and the $$d$$-connectivity of the graph will be unchanged. We refer the reader to the textbook of Koller and Friedman for a full proof.
