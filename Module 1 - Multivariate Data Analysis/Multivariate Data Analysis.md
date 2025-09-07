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
