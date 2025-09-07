## Multivariate Normal Distribution
It generalizes the normal distribution to multiple variables.

Definition: A vector $X = (X_1, X_2, ..., X_n)$ follows a multivariate normal distribution if every linear combination of its components has a univariate normal distribution.

$X∼ N(\mu,\boldsymbol{\Sigma})$ 
Where:
- $\mu$: Mean vector (n×1)
- $\\boldsymbol{\Sigma}$: Covariance matrix (n×n)

## 1. Univariate Normal Distribution
The probability density function (PDF) of a univariate normal distribution with mean $\mu$ and variance $\sigma^2$ is:

\$$
\begin{aligned}
f(x) &= \frac{1}{\sqrt{2\pi \sigma^2}} \cdot \exp\left( -\frac{1}{2} \cdot \frac{(x - \mu)^2}{\sigma^2} \right)
            &= \frac{1}{\sqrt{2\pi \sigma^2}} \cdot \exp\left( -\frac{1}{2} \cdot (x - \mu)(\sigma^2)^{-1}(x-\mu) \right)
\end{aligned}
\$$


This is the squared standardized distance from the mean. The variance $\sigma^2$ shows the spread of data.

## 2. Bivariate Normal Distribution

Let $X_1$ and $X_2$ be independent normal variables with means $\mu_1$, $\mu_2$ and variances $\sigma_1^2$, $\sigma_2^2$, respectively. 

Then the joint PDF is:

$$
\begin{aligned}
f(x_1, x_2) &= f(x_1) \cdot f(x_2) \\
&= \frac{1}{\sqrt{2\pi \sigma_1^2}} \cdot \exp\left( -\frac{1}{2} \cdot \frac{(x_1 - \mu_1)^2}{\sigma_1^2} \right) \cdot \\ \frac{1}{\sqrt{2\pi \sigma_2^2}} \cdot \exp\left( -\frac{1}{2} \cdot \frac{(x_2 - \mu_2)^2}{\sigma_2^2} \right) \\
&= \frac{1}{2\pi \sigma_1 \sigma_2} \cdot \exp\left( -\frac{1}{2} \left[
\frac{(x_1 - \mu_1)^2}{\sigma_1^2} +
\frac{(x_2 - \mu_2)^2}{\sigma_2^2}
\right] \right)
\end{aligned}
$$



Let us define the vectors:

\$$
\mathbf{X} =
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}, \quad
\boldsymbol{\mu} =
\begin{bmatrix}
\mu_1 \\
\mu_2
\end{bmatrix}
\$$

Then the difference vector is:

\$$
\mathbf{X} - \boldsymbol{\mu} =
\begin{bmatrix}
x_1 - \mu_1 \\
x_2 - \mu_2
\end{bmatrix}
\$$

For a dataset with two (or more) features, the covariance matrix captures how each variable varies with every other. 


(The correlation of variables $x_1$ and $x_2$ is defined by $\rho(x_1,x_2)=\frac{cov(x_1,x_2)}{\sigma_1 \cdot \sigma_2}$. Then the covariance becomes $cov(x_1,x_2)=\rho \sigma_1 \sigma_2$)

$$\boldsymbol{\Sigma} = \begin{bmatrix}
var(x_1) & cov(x_1,x_2) \\
cov(x_2,x_1) & var(x_2)
\end{bmatrix} = 
\begin{bmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{bmatrix}$$

### Case 1: Without correlation $(\rho=0)$  - Variables are independent
![image](https://github.com/user-attachments/assets/a95b2ef7-73a7-477f-b47a-bcf2bec50e5e)![image](https://github.com/user-attachments/assets/e7d07ed4-0693-433d-ba3a-347c7c79bddb)


Then, the covariance matrix becomes diagonal:

$$
\boldsymbol{\Sigma} = \begin{bmatrix}
var(x_1) & cov(x_1,x_2) \\
cov(x_2,x_1) & var(x_2)
\end{bmatrix} = 
\begin{bmatrix}
\sigma_1^2 & 0 \\
0 & \sigma_2^2
\end{bmatrix}
$$

Compute the Determinant of the Covariance Matrix 

\$$
\left| \boldsymbol{\Sigma} \right| =
\left| 
\begin{array}{cc}
\sigma_1^2 & 0 \\
0 & \sigma_2^2
\end{array}
\right| 
= \sigma_1^2 \sigma_2^2
\$$

Compute the Inverse of the Covariance Matrix

$$
\begin{aligned}
\boldsymbol{\Sigma}^{-1} &= \frac{1}{\left| \boldsymbol{\Sigma} \right|} Adj(\boldsymbol{\Sigma} \)
\\
&=\begin{bmatrix}
\sigma_1^2 & 0 \\
0 & \sigma_2^2
\end{bmatrix}^{-1}
\\
&= \frac{1}{\sigma_1^2 \sigma_2^2} \begin{bmatrix}
\sigma_2^2 & 0 \\
0 & \sigma_1^2
\end{bmatrix}
\\
&=\begin{bmatrix}
\frac{1}{\sigma_1^2} & 0 \\
0 & \frac{1}{\sigma_2^2}
\end{bmatrix}\\
\end{aligned}
$$


Now compute the quadratic form:

\$$
\begin{aligned}
(\mathbf{X} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu})
&= \begin{bmatrix}
x_1 - \mu_1 & x_2 - \mu_2
\end{bmatrix}
\begin{bmatrix}
\frac{1}{\sigma_1^2} & 0 \\
0 & \frac{1}{\sigma_2^2}
\end{bmatrix}
\begin{bmatrix}
x_1 - \mu_1 \\
x_2 - \mu_2
\end{bmatrix}
\\
&= \begin{bmatrix}
x_1 - \mu_1 & x_2 - \mu_2
\end{bmatrix}
\begin{bmatrix}
\frac{x_1 - \mu_1}{\sigma_1^2} \\
\frac{x_2 - \mu_2}{\sigma_2^2}
\end{bmatrix}
\\
&=\begin{bmatrix}
\frac{(x_1 - \mu_1)^2}{\sigma_1^2}+\frac{(x_2 - \mu_2)^2}{\sigma_2^2}
\end{bmatrix}\\
\end{aligned}
$$


Then the joint PDF is:

$$f(x_1, x_2)
= \frac{1}{(2\pi)^{(2/2)} \cdot \left| \boldsymbol{\Sigma} \right|^{1/2}} \cdot \exp\left( -\frac{1}{2} (\mathbf{X} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \right)
 = \frac{1}{2\pi \sigma_1 \sigma_2} \cdot \exp\left( -\frac{1}{2} \left[
\frac{(x_1 - \mu_1)^2}{\sigma_1^2} +
\frac{(x_2 - \mu_2)^2}{\sigma_2^2}
\right] \right)
$$

### Case 2: With correlation $(\rho)$ - Variables are not independent
![image](https://github.com/user-attachments/assets/c4932c56-e85d-4cdc-8690-51d601c23d62) ![image](https://github.com/user-attachments/assets/4900ce76-dd5b-4bee-99b9-2af3d248a197)


$$\boldsymbol{\Sigma} = 
\begin{bmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{bmatrix}$$

Compute the Determinant of the Covariance Matrix 

\$$
\begin{aligned}
\left| \boldsymbol{\Sigma} \right| &=
\left| 
\begin{array}{cc}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{array}
\right| \\
&= \sigma_1^2 \sigma_2^2 - \rho^2 \sigma_1^2 \sigma_2^2 \\
&=  \sigma_1^2 \sigma_2^2 (1-\rho^2)
\end{aligned}
\$$

Compute the Inverse of the Covariance Matrix

$$
\begin{aligned}
\boldsymbol{\Sigma}^{-1} &= \frac{1}{\left| \boldsymbol{\Sigma} \right|} Adj(\boldsymbol{\Sigma} \)
\\
&=\begin{bmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{bmatrix}^{-1}
\\
&= \frac{1}{\sigma_1^2 \sigma_2^2 (1-\rho^2)} 
\begin{bmatrix}
\sigma_2^2 & -\rho \sigma_1 \sigma_2 \\
-\rho \sigma_1 \sigma_2 & \sigma_1^2
\end{bmatrix}
\\
&= \frac{1}{(1-\rho^2)}
\begin{bmatrix}
\frac{1}{\sigma_1^2} & \frac{-\rho}{ \sigma_1 \sigma_2} \\
\frac{-\rho}{ \sigma_1 \sigma_2} & \frac{1}{\sigma_2^2}
\end{bmatrix}\\
\end{aligned}
$$

Now compute the quadratic form:

\$$
\begin{aligned}
(\mathbf{X} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu})
&= \begin{bmatrix}
x_1 - \mu_1 & x_2 - \mu_2
\end{bmatrix}
 \frac{1}{(1-\rho^2)}
\begin{bmatrix}
\frac{1}{\sigma_1^2} & \frac{-\rho}{ \sigma_1 \sigma_2} \\
\frac{-\rho}{ \sigma_1 \sigma_2} & \frac{1}{\sigma_2^2}
\end{bmatrix}
\begin{bmatrix}
x_1 - \mu_1 \\
x_2 - \mu_2
\end{bmatrix}
\\
&= \begin{bmatrix}
x_1 - \mu_1 & x_2 - \mu_2
\end{bmatrix}
\begin{bmatrix}
\frac{x_1 - \mu_1}{\sigma_1^2}- \frac{\rho(x_2 - \mu_2)}{ \sigma_1 \sigma_2} \\
\frac{x_2 - \mu_2}{\sigma_2^2} - \frac{\rho(x_1 - \mu_2)}{ \sigma_1 \sigma_2}
\end{bmatrix}
\\
&=\begin{bmatrix}
\frac{(x_1 - \mu_1)^2}{\sigma_1^2}+\frac{(x_2 - \mu_2)^2}{\sigma_2^2} - \frac{\rho(x_1 - \mu_1)(x_2 - \mu_2)}{ \sigma_1 \sigma_2} - \frac{\rho(x_1 - \mu_1)(x_2 - \mu_2)}{ \sigma_1 \sigma_2}
\end{bmatrix}\\
&=\begin{bmatrix}
\frac{(x_1 - \mu_1)^2}{\sigma_1^2}+\frac{(x_2 - \mu_2)^2}{\sigma_2^2} - 2\frac{\rho(x_1 - \mu_1)(x_2 - \mu_2)}{ \sigma_1 \sigma_2} 
\end{bmatrix}\\
\end{aligned}
$$

Then the joint PDF is:

$$f(x_1, x_2)
 = \frac{1}{2\pi \sigma_1 \sigma_2 \(1-\rho^2)^{1/2}} \cdot \exp\left( -\frac{1}{2} \begin{bmatrix}
\frac{(x_1 - \mu_1)^2}{\sigma_1^2}+\frac{(x_2 - \mu_2)^2}{\sigma_2^2} - 2\frac{\rho(x_1 - \mu_1)(x_2 - \mu_2)}{ \sigma_1 \sigma_2} 
\end{bmatrix} \right)
= \frac{1}{(2\pi)^{(2/2)} \cdot \left| \boldsymbol{\Sigma} \right|^{1/2}} \cdot \exp\left( -\frac{1}{2} (\mathbf{X} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \right)
$$

## 3. Generalization to the Multivariate Case
![image](https://github.com/user-attachments/assets/e7d9cd3d-9595-4754-95df-1d84bd62b861)

Let $\mathbf{X} \in \mathbb{R}^n$ be a random vector following a multivariate normal distribution with mean vector $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$.

\$$
\mathbf{X} =
\begin{bmatrix}
X_1 \\
X_2 \\
\vdots \\
X_n
\end{bmatrix}, \quad
\boldsymbol{\mu} =
\begin{bmatrix}
\mu_1 \\
\mu_2 \\
\vdots \\
\mu_n
\end{bmatrix}
\$$

\$$
\boldsymbol{\Sigma} =
\begin{bmatrix}
\text{Var}(X_1) & \text{Cov}(X_1,X_2) & \cdots & \text{Cov}(X_1,X_n) \\
\text{Cov}(X_2,X_1) & \text{Var}(X_2) & \cdots & \text{Cov}(X_2,X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_n,X_1) & \cdots & \cdots & \text{Var}(X_n)
\end{bmatrix}
= \begin{bmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 & \cdots & \rho \sigma_1 \sigma_n \\
\rho \sigma_2 \sigma_1 & \sigma_2^2 & \cdots & \rho \sigma_2 \sigma_n \\
\vdots & \vdots & \ddots & \vdots \\
\rho \sigma_n \sigma_1 & \rho \sigma_n \sigma_2 & \cdots & \sigma_n^2
\end{bmatrix}
\$$


### Final PDF of the Multivariate Normal Distribution with Independent Variables

\$$
f(\mathbf{X}) = \frac{1}{(2\pi)^{n/2} |\boldsymbol{\Sigma}|^{1/2}} \cdot \exp\left( -\frac{1}{2} (\mathbf{X} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{X} - \boldsymbol{\mu}) \right)
\$$


# Properties of Multivariate Normal Distribution

Let $X_{n \times 1} \sim \mathcal{N}_n(\mu, \Sigma)$, i.e., a multivariate normal random vector with mean vector $\mu$ and covariance matrix $\Sigma$. Then the following properties hold:

---

## 1. Each Variable is Univariate Normal
If the whole vector $X$ is multivariate normal, then each component $X_j$ is also normally distributed.

If $X \sim \mathcal{N}_n(\mu, \Sigma)$, then for all $j = 1, 2, \dots, n$:   $X_j \sim \mathcal{N}(\mu_j, \sigma_j^2)$

**Example**:  
If  

$$
X = \begin{bmatrix} 
X_1 \\ 
X_2 
\end{bmatrix} 
\sim \mathcal{N}_2\left( 
\begin{bmatrix} 
5 \\ 
3 
\end{bmatrix},\ 
\begin{bmatrix} 
4 & 1 \\ 
1 & 2 \end{bmatrix} 
\right)
$$


Then:  
- $X_1 \sim \mathcal{N}(5, 4)$  
- $X_2 \sim \mathcal{N}(3, 2)$  

---

## 2. Any Subset is Multivariate Normal

If you select a few components from the vector $X$, the resulting subset is also multivariate normal.
  
If $X_{n \times 1} \sim \mathcal{N}_n(\mu, \Sigma)$, then for any subset $X_q$:  $X_q \sim \mathcal{N}_q(\mu_q, \Sigma_q)$

**Example**:  
From a 5-variable MVN vector, selecting $X_1$, $X_3$, and $X_5$ results in:

$$
X_q = \begin{bmatrix} X_1 \\ X_3 \\ X_5 \end{bmatrix} \sim \mathcal{N}_3(\mu_q, \Sigma_q)
$$

---

## 3. Any Linear Combination is Univariate Normal

If you make a linear combination like $Y = a_1X_1 + a_2X_2 + \dots + a_nX_n$, then $Y$ will follow a normal distribution.
  
If $X \sim \mathcal{N}_n(\mu, \Sigma)$ and $a \in \mathbb{R}^n$, then:

$$
Y = a^T X \sim \mathcal{N}(a^T \mu, a^T \Sigma a)
$$

**Example**:  
Using the same $X$ as above:

$$Y = 2X_1 - 3X_2 \sim \mathcal{N}(2 \cdot 5 - 3 \cdot 3, \, 2^2 \cdot 4 + (-3)^2 \cdot 2 + 2 \cdot 2 \cdot (-3) \cdot 1)$$  
So,  
$Y \sim \mathcal{N}(1, 16 + 18 - 12 = 22)$

---

## 4. Linear Transformation is Multivariate Normal

If we apply a matrix $A$ to $X$, the result is still a multivariate normal distribution.

If $X \sim \mathcal{N}_n(\mu, \Sigma)$ and $A$ is a $q \times n$ matrix, then:

$$
Y = AX \sim \mathcal{N}_q(A\mu, A\Sigma A^T)
$$

**Example**:  
If $X$ is a 3D MVN vector and you define two new variables $Y_1, Y_2$ using a matrix $A$, then:

$$
Y = \begin{bmatrix} Y_1 \\ Y_2 \end{bmatrix} = A X \sim \mathcal{N}_2(A \mu, A \Sigma A^T)
$$

---

## 📘 Problems Based on Multivariate Normal Distribution

---

### 🔹 Q1

Let $X \sim \mathcal{N}_3(\mu, \Sigma)$ where:

$$
\mu = \begin{bmatrix} 5 \\ 3 \\ 7 \end{bmatrix}, \quad
\Sigma = \begin{bmatrix}
4 & -1 & 0 \\
-1 & 2 & 2 \\
0 & 2 & 9
\end{bmatrix}
$$

#### a) Find $P(X_1 > 6)$

Since $X \sim \mathcal{N}_3(\mu, \Sigma)$ then using the property (If the whole vector $X$ is multivariate normal, then each component $X_j$ is also normally distributed): 
$X_1 \sim \mathcal{N}(\mu=5, \sigma=4)$

So,

$$
\begin{aligned}
P(X_1 > 6) &= P(\frac{X_1 - \mu}{\sigma} > \frac{6 - 5}{2})\\
&= P(Z > 0.5) \\
&= 1- P(Z<=0.5) \\
&= 1 - 0.6915 \\
&= \boxed{0.3085} \\
\end{aligned}
$$

---

#### b) Find $P(5X_2 + 4X_3 > 70)$

Let $Y = 5X_2 + 4X_3$

If you make a linear combination like $Y = a_1X_1 + a_2X_2 + \dots + a_nX_n$, then $Y$ will follow a normal distribution.
Then $Y \sim \mathcal{N}(\mu', \sigma')$.

First, compute the expected value:

$$
\begin{aligned}
\mathbb{E}(Y) &= \mathbb{E}(5X_2 + 4X_3) \\
&= 5 \mathbb{E}(X_2) + 4 \mathbb{E}(X_3)\\
&=  5 \cdot 3 + 4 \cdot 7 \\
&= 15 + 28 \\
&= 43
\end{aligned}
$$

Now compute variance:
From $\Sigma$: $\text{Var}(X_2) = 4$, $\text{Var}(X_3) = 9$, $\text{Cov}(X_2, X_3) = 2$

$$
\begin{aligned}
\text{Var}(Y) &= \text{Var}(5X_2 + 4X_3) \\
&= 5^2 \cdot \text{Var}(X_2) + 4^2 \cdot \text{Var}(X_3) + 2 \cdot 5 \cdot 4 \cdot \text{Cov}(X_2, X_3) \\
&= 25 \cdot 4 + 16 \cdot 9 + 40 \cdot 2 \\
&= 100 + 144 + 80 \\
&= 324
\end{aligned}
$$

Then $Y \sim \mathcal{N}(\mu'=43, \sigma'=324)$.

Now standardize:

$$
\begin{aligned}
P(Y> 70)&=P(\frac{Y - 43}{\sqrt{324}} > \frac{70 - 43}{\sqrt{324}}) \\
&= P(Z > 1.5) \\
&= 1-P(Z<=1.5) \\
&= 1-0.9332\\
&=  \boxed{0.0668}
\end{aligned}
$$

---

#### c) Find $P(4X_1-3X_2 + 5X_3 < 80)$
Let $Y = 4X_1-3X_2 + 5X_3$

If you make a linear combination like $Y = a_1X_1 + a_2X_2 + \dots + a_nX_n$, then $Y$ will follow a normal distribution.
Then $Y \sim \mathcal{N}(\mu', \sigma')$.

First, compute the expected value:

$$
\begin{aligned}
\mathbb{E}(Y) &= \mathbb{E}(4X_1-3X_2 + 5X_3) \\
&= 4 \mathbb{E}(X_1)- 3 \mathbb{E}(X_2) + 5 \mathbb{E}(X_3)\\
&= 4 \cdot 5- 3 \cdot 3 + 5 \cdot 7 \\
&= 20-9 + 35 \\
&= 46
\end{aligned}
$$

Now compute variance:
From $\Sigma$: $\text{Var}(X_1) = 4$, $\text{Var}(X_2) = 4$, $\text{Var}(X_3) = 9$, $\text{Cov}(X_1, X_2) = -1$, $\text{Cov}(X_2, X_3) = 2$, $\text{Cov}(X_1, X_3) = 0$

$$
\begin{aligned}
\text{Var}(Y) &= \text{Var}(4X_1-3X_2 + 5X_3) \\
&= 4^2 \cdot \text{Var}(X_1) +3^2 \cdot \text{Var}(X_2) + 5^2 \cdot \text{Var}(X_3) + 2 \cdot 4 \cdot (-3) \cdot \text{Cov}(X_1, X_2)+ 2 \cdot (-3) \cdot 5 \cdot \text{Cov}(X_2, X_3)+ 2 \cdot 4 \cdot 5 \cdot \text{Cov}(X_1, X_3) \\
&= 16 \cdot 4 + 9 \cdot 4 + 25 \cdot 9 - 24 \cdot (-1)- 30 \cdot 2 + 40 \cdot 0  \\
&= 64 + 36+225+24-60 + 0 \\
&= 289
\end{aligned}
$$

Then $Y \sim \mathcal{N}(\mu'=46, \sigma'=289)$.

Now standardize:

$$
\begin{aligned}
P(Y< 80)&=P(\frac{Y - 46}{\sqrt{289}} < \frac{80 - 46}{\sqrt{289}}) \\
&= P(Z < 2) \\
&=  \boxed{0.9773}
\end{aligned}
$$

---

### 🔹 Q2

Let $X$ and $Y$ be jointly normal with:

- $\mu_X = 1$, $\sigma_X^2 = 1$
- $\mu_Y = 0$, $\sigma_Y^2 = 1$
- $\rho = \frac{1}{2}$

Find $P(2X + Y < 3)$

Let $V = 2X + Y$. Then:

$$
\mathbb{E}(V) = 2 \cdot \mathbb{E}(X) + \mathbb{E}(Y) = 2 \cdot 1 + 0 = 2
$$

Since $\text{Cov}(X,Y) = \rho \cdot \sigma_X \cdot \sigma_Y = \frac{1}{2} \cdot 1 \cdot 2 = 1$

$$
\text{Var}(V) = 2^2 \cdot \text{Var}(X) + \text{Var}(Y) + 2 \ cdot 2 \cdot 1 \cdot \text{Cov}(X, Y)
= 4 + 4 + 4 \cdot 1 = 4 + 4 + 4 = 12
$$

Then,

$$
V \sim \mathcal{N}(2, 12)
$$

Now standardize:

$$
\begin{aligned}
P(V< 3)&=P(\frac{V - 2}{\sqrt{12}} < \frac{3 - 2}{\sqrt{12}}) \\
&= P(Z < \frac{1}{\sqrt{12}}) \\
&= P(Z < 0.2886) \\
&=  \boxed{0.6103}
\end{aligned}
$$

---
