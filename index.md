---
layout: post
title: 目录
---
这些笔记构成一个关于概率图模型的简洁的初级课程{% sidenote 1 '概率图模型是机器学习的子领域，研究如何用概率对事件进行描述与推断。'%}.
这些笔记基于史丹佛大学[CS228](http://cs.stanford.edu/~ermon/cs228/index.html)课程，由[Stefano Ermon](http://cs.stanford.edu/~ermon/)执教，由[Volodymyr Kuleshov](http://www.stanford.edu/~kuleshov)在许多学生与助教的[帮助下](https://github.com/ermongroup/cs228-notes/commits/master)完成。
{% marginnote 'mn-id-whatever' '这些笔记 **还在建设中**！虽然我们已经写好大部分的内容，但其中可能包含错误。如果你发现错误，请告诉我们或是提交pull request到我们的[Github仓库](https://github.com/ermongroup/cs228-notes).'%}
你也可以把你的改进用[Github](https://github.com/ermongroup/cs228-notes)提交给我们。

这个课程首先介绍概率图模型的基础，最后会从头讲解[变自编码器]()，这是一个很重要的概率模型，同时也是近年来深度学习中最重要的成果之一。

## 前置

1. [导论](preliminaries/introduction/)：概率图模型是什么？综览本课程。

2. [复习概率论](preliminaries/probabilityreview)：概率分布、条件概率、随机变量。

3. [实际应用范例](preliminaries/applications)：图形降燥、预测RNA构造、句法分析、光学字体辨识(OCR)。

## 表示

1. [贝叶斯网络](representation/directed/)：定义、有向图表示。有向模型中的独立性。

2. [马可夫网络](representation/undirected/)：无向模型与有向模型比较。无向模型中的独立性。条件随机场(CRF)。

## 推断

1. [消除变量法](inference/ve/) 推断问题。消除变量法。推断过程的复杂度。

2. [信念传播方程](inference/jt/)：联合树算法。任意图模型中进行精确推断。带循环的信念传播过程。

3. [最大后验概率推断](inference/map/)：Max-sum信息传播。Graphcuts。线性规划条件松弛。偶分解方法。

4. [抽样推断](inference/sampling/)：蒙地卡罗抽样法。重要性抽样。马可夫链蒙地卡罗法。推断中的应用。

5. [变量推断](inference/variational/)：Variational lower bounds、Mean Field、Marginal polytope与其条件松弛。

## 学习

1. [有向模型中的学习](learning/directed/)：最大似然估计。学习理论基础。贝叶斯网络中的最大似然估计。

2. [无向模型中的学习](learning/undirected/)：Exponential families、用梯度降下法进行最大似然估计。CRF中的学习。

3. [隐函数模型中的学习](learning/latent/)：隐函数模型。混和高斯模型。预期最大化。

4. [贝叶斯学习](learning/bayesianlearning/)：贝叶斯典范。共轭先验分配。例子。

5. [结构学习](learning/structLearn/)：周刘算法、赤池信息量准则、贝叶斯信型量准则、贝叶斯结构学习。

## 融会贯通

1. [变自编码器](extras/vae)：一种深度生成模型、重新赋予参数的技巧。学习隐含的视觉表现。

2. 延伸阅读列表：结构性支持向量机。贝叶斯的非参数性模型。

## 关于翻译

[译名对照表](extras/glossary)
