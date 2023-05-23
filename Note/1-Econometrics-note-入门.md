# 计量入门

写在最前面

## 1.引言

#### 参考书

1. Bruce.Hansen，Econometrics. 2020
2. Jeffreu.M.Wooldridge, Econometric Analysis of Cross section and Panel Data. MIT.2010

#### 什么是计量经济学

##### 标准定义

计量经济学是以下三个东西的有机结合
1. 经济模型
2. 数量统计
3. 经济数据

##### 4大部分

1. 模型形式
	- 线性模型、广义线性模型、分位数模型、...
2. 参数估计
	- OLS、ML、GMM、...
3. 参数估计的假设检验
	-  CLT 中心极限定理
4. 实际应用

#### 理论基础

1. 线性代数
	- 矩阵的运算
		- 矩阵理论
2. 高等数学
	- 泰勒展开
		- 实分析
		- 渐近理论
3.  概率论
	- 测度论

#### 软件

- R
- Matlab
- Stata
- Python

## 2.条件期望和预测
---

### 2.1.要解决的问题
假设我们要定量的研究工资水平。那么我可以把问题分解成几个逐步深入的小问题：
 1.全国工资水平到底怎样，如何度量工资水平
 2.不同的分类标准下，工资水平怎么分别度量

> 最常用的计量经济学工具是最小二乘估计，也称为回归。最小二乘法被用于估计一个变量（因变量、回归变量、条件变量或协变量）的条件均值。

![[Pasted image 20230523202424.png]]


### 2.2.需要了解的基本概念
回顾如下概念和写法,后面都要用到：

|中文 |英文|公式|
|---|---|---|
|概率分布|probability distribution|$F(u) = \Bbb P[wage \le u]$|
|概率密度函数|probability density function|$f(u)=\frac{d}{du}F(u)$|
|中位数|median|$F(m)=\frac{1}{2}$时的$m$|
|均值/预期|mean/expectation|$$\mu=\Bbb E[Y]=\sum_{j=1}^\infty\tau_j\Bbb P[Y=\tau_j]$$$$\mu=\Bbb E[Y]=\int_{-\infty}^\infty yf(y)dy$$|
|分位点|quantiles||
|百分位点|percentiles||

对工资分布进行描述时，最重要的概念是均值（mean）。

> #NOTICE  **为什么是均值而不是中位数？**
> 
> - 中位数是集中趋势的一个稳健的度量，但是它很难用于计算，因为不是一个线性算子
> - 期望是对集中趋势的一个方便的度量，因为它是一个线性算子，但是它不稳健
> > #NOTICE  **什么叫做稳健？** 就是对分布的尾部敏不敏感

### 2.3.条件预期 Conditional Expectation 

不同的性别、人种，人们的工资水平不同。性别和人种的变化是有限个数的，离散变量。

**条件期望**是给定条件时的因变量的期望(或平均值），表示为：$\Bbb E[Y|X=x]$

条件期望最大的优点是它将复杂的分布简化为单一的汇总度量，从而促进组间的比较。也正是由于这种性质，条件均值/预期是回归分析的主要兴趣点，也是计量经济学的主要焦点。

### 2.4.对数与百分比数 Logs and Percentages

在图2.1(c)中，对数据进行取对数的操作。这张图看起来更近似于正态分布，尖峰厚尾的情况不那么明显。

> #NOTICE  **为什么要用对数？**
> 
>1. 当应用于数字时，对数的差近似等于两个数的百分数差
>	1. 比较a、b这两个正数的百分数差
>	2. 百分数差表示为：$p = 100(\frac{a-b}{b})$
>	3. 变换：$\frac{a}{b}=1+\frac{p}{100}$
>	4. 取对数：$\log a- \log b = \log (1+\frac{p}{100})$
>	5. 因为对数有个性质：$\log (1+x) \approx x$
>	6. 所以：$p \approx 100(\log a -\log b)$
>2. 当应用于平均时，对数的差近似等于几何平均值的百分比差异

### 2.5.条件期望函数 Conditional Expectation Function

教育水平是工资的重要影响因素。而教育水平可以用受教育的年数来衡量，离散变量

用数学符号来表示：
- 将性别、人种和受教育年数等多个条件变量记为：$X_1,X_2,\dots ,X_k$
- 将工资的对数$\log(wages)$这个因变量记为：$Y$

条件预期就可以写成：$$\Bbb E[Y|X_1=x_2,X_1=x_2,\dots,X_k=x_k]=m(x_1,x_2,\dots,x_k)$$
上面这个式子被称为**条件期望函数 conditional expectation function (CEF)**。

用向量的形式表达就是：$$\Bbb E[Y|X=x]=m(x)$$
其中X是一个k行1列的向量：$$X = \begin{pmatrix}X_1 \\ X_2 \\  \dots \\ X_k\end{pmatrix}$$
注意：
- $X_i$表示的是条件
- $x_i$表示这个条件对应的取值

### 2.6 连续变量 Continuous Variables

前面的条件都是离散的，但是实际上有很多条件变量是连续，该如何表示？
$$m(x)=\Bbb E[Y|X=x]=\int_{-\infty}^{\infty}yf_{Y|X}(y|x)dy$$

#todo  

JDF：Joint Distribution Function 联合分布函数

JPDF：Joint Probobility Density Function 联合概率密度函数

MDF：marginal density Function 边缘概率密度 


### 2.7 重复期望法则Law of Iterated Expectations

条件期望的期望是无条件的期望。换句话说，条件平均值的平均值是无条件的平均值

> ###### 定理2.1 简单重复期望法则
> 
> - 如果$E|Y|<\infty$，那么对于任意随机向量$X$有：$$\Bbb E[\Bbb E[Y|X]]=\Bbb E[Y]$$

- 离散变量X：$$\Bbb E[\Bbb E[Y|X]]=\sum_{j=1}^{\infty}\Bbb E[Y|X=x_j]\Bbb P[X=x_j]$$
对条件变量的向量中的所有条件变量，分别计算$Y$的条件预期，然后再将所有$Y$的条件预期求和。实际上就是穷尽所有可能情况，并计算Y的条件期望之和，那也就是Y的预期。

- 连续变量X：$$\Bbb E[\Bbb E[y|x]]=\int_{\Bbb R^k}\Bbb E[Y|X=x]f_X(x)dx$$
> ###### 定理2.2 重复期望法则
> 
> - 如果$E|Y|<\infty$，那么对于任意随机向量$X_1$和$X_2$有：$$\Bbb E[\Bbb E[Y|X_1,X_2]|X_1]=\Bbb E[Y|X_1]$$

条件期望的一个属性是，当你对一个随机向量X提出条件时，你可以把它当作常数。因此：

> ###### 定理2.3 条件定理 Conditioning Theorem
> 
> - 如果$E|Y|<\infty$，有：$$\Bbb E[g(X)Y|X]=g(X)\Bbb E[Y|X]$$
> - 如果$E|g(X)|<\infty$，有：$$\Bbb E[g(X)Y]=\Bbb E[g(X)\Bbb E[Y|X]]$$









## 2.数学基础

### 2.1.总体与样本

- 总体population
	- 客观存在，但有无法完全获得
- 样本samples
	- 你观察到的数据

> 案例
> 1. 计算全球人口的平均身高
> 	总体（未知）：全球人口
> 	样本（可得）：随即抽出来的人 
> 2. 抛硬币
> 	总体（未知）：抛无穷次
> 	样本（可得）：抛到次硬币
> 3. 得到袋子里红球的比例
> 	袋子里有无穷个求，红球的比例是一定的
> 	总体（未知）：袋子中红球的比例
> 	样本（可得）：抽100个球，用红球的比例去近似的推断总体的红球概率

大数定律

#### 总体和样本精讲

> 例子1：用直方图近似N(0,1)的PDF
> - 从N(0,1)随机抽取100个数

> 例子2：抽取18岁人类的身高数据
> - 100个人的数据
> - 10000个人的数据

| 名称 |特点 |区别|
|---|:---|:---:|
|总体|1.本质上是一个分布<br>2.具有无限的数据<br>3.通常未知|人不知道，上帝视角才知道|
|样本|1.本质是你手上的数据<br>2.有限个数据<br>3.不同的人手上的样本可能不一样|只能获得部分信息，是总体的一个子集|
那么：
1. 用样本的性质逼近总体的性质
2. 如果知道总体分布具体表达式，就可以知道总体的所有信息

### 2.2.条件概率

P(A)：A事件发生的概率

条件概率：
$$P(A|B)=\frac{P(AB)}{P(B)}$$
- $P(B)$
- $P(A|B)$
- $P(AB)$

> ###### 案例
> 
> 一个不透明的袋子有2个红球，1个黑球
> - A事件 = 抽出一个红球
> - B事件 = 抽出一个黑球
> - A|B = 已经抽出一个黑球的情况下，抽出来一个红球
> - AB = 第一次抽是红球，第二次抽出来是黑球

$P(A) = 2/3$
$P(B) = 1/3$
$P(AB) = \frac{2}{3}\times\frac{1}{2}$

> (r1,b),(r1,r2),(r2,r1),(r2,b),(b,r1),(b,r2)  

特殊情况：如果AB事件独立
P(AB) = P(A)P(B)

### 2.3.随机变量

X，Y：随机变量。random variable, r.v.

CDF：cumulative distribution function 累计分布函数

PDF：probobility Density Function 概率密度函数



### 2.4.条件期望函数

**CEF：Condition Expectation Function** 

$E[Y|X=x] = m(x)$
- $X,Y, r.v.$
- Y是X的函数，比如$m(x)=x$ , $m(x)=x^2$, $m(x)=log(x)$，...
	- 当x取某一个值时，都有唯一的y与之对应

**期望迭代法则** Law of iterated expectation
- 如果$E|y|<\infty$，那么对于任意随机向量$x$有：$$E[E[y|x]]=E[y]$$

**条件定律**：Conditioning Theorem




## 3. 条件期望与预测

### 3.1. 条件期望函数误差

**Conditional Expectation Function Error**是一个<font style="color:#ff3737">定义式</font>。
- CEF: $m(x)=E[Y|X=x]$
- CEFE: $e=Y-m(x)=Y-E[Y|X=x]$
- 其中：
	- Y，X r.v.

**性质**：
- $E[e|x]=0$
- $E[e]=0$
- $E[e · h(x)]=0$

**特例**：
1. 截距模型：$m(x) = E[Y]=\mu$ 
$$Y=\mu + e$$
2. 线性模型：$m(x)=\beta X$  
$$Y = \beta X+e$$
3. Logit模型：$m(x)=\frac{e^{x\beta}}{1+e^{x\beta}}=E[Y|x]$
$$Y = \frac{e^{x\beta}}{1+e^{x\beta}} + e$$
### 3.2. 回归模型方差

**Regression Variance**
$$\sigma^2=E[(e-E[e])^2]=E[e^2] = Var(e)$$
**含义**：
- $\sigma^2$ 衡量了因变量中无法被自变量解释的部分的大小

**性质**：
-  $\sigma^2$ 随自变量个数的增加而减小

**定理**：$m(x)$是最好的因变量的预测值
- 假设$Ey^2<\infty$，那么对于任意函数$g(x)$，有$$E[(y-g(x))^2] \ge E[(y-m(x))^2] $$
-  其中：$m(x)=E[Y|x]$

**推导**：
$$
\begin{equation}
\begin{split}
E[(y-g(x))^2]=&E[(e+m(x)-g(x))^2] \\
=&E[e^2]+2E[e(m(x)-g(x))] + E[(m(x)-g(x))^2]\\
=&E[e^2]+E[(m(x)-g(x))^2]\\
\ge&E[e^2]\\
=&E[(y-m(x))^2]
\end{split}
\end{equation}
$$

### 3.3.同方差和异方差

**条件方差：Conditional Variance**
$$
\begin{equation}
\begin{split}
\sigma^2=&var(y|x) \\
=&E[(y-E[y|x)^2|x]\\
=& E[e^2|x)]\\
=&\sigma^2(x)
\end{split}
\end{equation}
$$
- **关系**
	- $E[e^2|x]$ = $\sigma^2$
- **一个有用的变换**
	- 定义：$\epsilon = \frac{e}{\sigma(x)}$，其中$\sigma(x)=\sqrt{\sigma^2}$
	- 性质：
		1.$E[\epsilon|e]=0$
		2.$var(\epsilon|x)=1$
- **条件方差的含义**
	- 条件方差刻画了Y在某个条件均值附近的随机波动范围

**同方差：homoskedasticity  homogeneity**
- $var(e|x)=C$
- $C$是一个常数，不随x的变化而变化

**异方差：Heteroskedasticity,Heterogeneity**
- $var(e|x)=\sigma^2(x)$
- 给定不同的x，y在对应的条件均值附近的随机波动范围不一样

> #NOTICE
> 1. 同方差是异方差的一个特例，实际的数据中很少有同方差的现象
> 2. 同方差可以简化理论上的一些计算
> 3. 无论是同方差还是异方差，他们都是建立在条件方差的基础上

### 3.4.线性条件期望函数模型
|变量|中文说法 |英文说法 |
|---|---|---|
|Y|因变量、被解释变量|dependent variable|
|X|自变量、解释变量|independent variable、covariate|
**Linear CEF Model**
考虑这样一个模型：$X=X_1, m(x)=X_1\beta_1$
$$Y=X_1\beta_1+e$$
其中：
- 条件1：$E[e|x_1] = 0$

**Homoskedastic Linear CEF Model**
再增加一个条件：
- 条件2：$E[e^2|x_1] = \sigma^2$

### 3.5.线性投影模型——Linear Projection Model

定理：$m(x)=E[Y|X=x]$是Y最好的预测值。
- m(x)定义在总体层面
- m(x)的形式未知，用线性模型$X'\beta$来预测$Y$

定义：**均方预测误Mean squared prediction error**
$$S(\beta)=E[(Y-X'\beta)^2]$$
- 含义：给定参数$\beta$，用$X'\beta$预测Y的方差

定义：**最佳线性预测值Best Linear Predictor**
$$
\begin{cases}
E[Y|X=x]=X'\beta_p \\ 
\beta_p = \mathop argmin_{b\in \mathbb R}S(b) \\
\end{cases}
$$
$b'=\mathop argmin_{b \in \Bbb R}S(b)$：在$b\in(-\infty,\infty)$的区间内，找到一个$b'$使得$S(b)$在$b'$处达到最小值。

定义：**预测误差Projection error**
$$e_p=y-x'\beta$$
定义：**Linear Projection Model**



## 4.最小二乘法

样本的数学表示

数据是独立同分布的iid

含义：


### 4.1.单变量最小二乘

$\hat S$ 惯用于表示估计值

OLS、LS都是方法

### 4.2.多变量的最小二乘

如果你的数据没有做特殊处理，最好在模型中，加一个常数项、截距项

### 4.3.线性回归模型


