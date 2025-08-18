# Discriminant Analysis

## Definition
> Discriminant Analysis is a supervised, multivariate classification technique used to predict group membership based on observed features.

> Discriminant Analysis is used when:
- The **dependent variable** is **categorical** (i.e., it represents group labels or categories).
- The **independent variables** are **interval** or **ratio scale** (quantitative variables).
> It is a technique to discriminate between two or more mutually exclusive and exhaustive groups on the basis of some explanatory variables.

> It finds a linear combination of predictor variables (like age, income, test scores) that best separates observations into predefined classes (like pass/fail, spam/not spam, diseased/healthy).
---

## **Multiple Discriminant Analysis (MDA)**

- Used when there are **more than two groups** (e.g., three or more categories in the dependent variable).
- LDA with **multiple discriminant functions**, one fewer than the number of groups.
- Helps in visualizing the **group separation** in higher-dimensional space.

**Example**:  
Classifying a product preference into **three brands** based on price sensitivity, quality perception, and advertisement response.

---
## Comparison: Discriminant Analysis vs ANOVA vs Regression


### Similarities

| Aspect | Discriminant Analysis (DA) | ANOVA | Regression |
|--------|-----------------------------|-------|------------|
| **Number of Dependent Variables** | One |One |One |
| **Number of Independent Variables** | Multiple |Multiple |Multiple |


### Differences

| Feature | Discriminant Analysis | ANOVA | Regression |
|--------|------------------------|-------|------------|
| **Dependent Variable Type** | **Categorical** (e.g., Pass/Fail, Class A/B/C) | **Continuous**, but grouped into **categories** | **Continuous** (e.g., height, sales) |
| **Independent Variables** | **Continuous** (interval/ratio scale) | **Categorical** (usually grouping variable) | **Continuous or categorical** |
| **Goal** | **Classify** cases into known groups | **Test** if group means differ | **Predict** the value of a response variable |
| **Output** | **Classification rule**, group membership | **F-ratio**, significance of mean differences | **Regression equation**, predicted value |
| **Number of Groups** | 2 or more categories in dependent variable | 2 or more groups in factor | Usually one dependent variable |


---
## Assumptions of Discriminant Analysis

To use Discriminant Analysis effectively and interpret results correctly, certain statistical assumptions must be satisfied.

| Assumption | Description | Why? |
|------------|-------------|-------------|
| 1. Sample Size | Sample size should be **at least 5 times** the number of independent variables **per group** | A small sample size can make covariance estimates unstable and result in poor classification performance|
| 2. Normality | Each of the **independent variables should be normally distributed** within each group of the dependent variable | DA uses probability density functions that assume normality to compute classification functions accurately |
| 3. Equal Variances | Variance-covariance matrices are the same for all groups | This assumption ensures **linear** separation between groups. If violated, **Quadratic Discriminant Analysis (QDA)** should be used |
| 4. No Outliers | Data should not contain extreme values |DA is **highly sensitive to outliers**, which can distort group means, covariances, and classification accuracy |
| 5. No Multicollinearity among the independent variables | Predictors should not be highly correlated | Highly correlated predictors can make the discriminant function unstable and hard to interpret. |
| 6. Mutually Exclusive Groups | The groups (categories of the dependent variable) must be **mutually exclusive** | Each subject or case must belong to **only one group**, not multiple. |
| 7. Correct Classification | Each case in the training data must be **correctly classified** into the known group. Training labels must be accurate |The DA model learns from the group labels provided; incorrect labeling misguides the function.  |



> Violating these assumptions can lead to poor model accuracy, invalid statistical tests, and incorrect interpretations.

---

## Hypothesis in Discriminant Analysis (DA)

### **Null Hypothesis (H₀)**:
> The group means of a set of independent variables for two or more groups are **equal**.  
(There is **no significant difference** between the groups based on the independent variables.)

Mathematically:

$$H_0: \mu_1 = \mu_2 = \dots = \mu_k$$

Where $\mu_k$ is the mean vector of the independent variables for group $k$.


### **Alternative Hypothesis (H₁)**:
> The group means are **not equal**.  
(At least one group is significantly different from the others in terms of the independent variables).

$$H_1: \mu_i \ne \mu_j \quad \text{for some } i \ne j$$

---


### Evaluating a Discriminant Analysis

Discriminant Analysis not only builds classification rules, but also:

- Tests for **significant differences** among groups.
- Measures **classification accuracy** (how well the model classifies cases into correct groups).
- Tells us **how many categories** the dependent variable contains and how well they are separated.

---

# 📘 Statistical Background for Discriminant Analysis

This document introduces essential statistical concepts used as the foundation for understanding **Discriminant Analysis**. 
The concepts are organized clearly with definitions and examples for easy understanding.

<img width="1894" height="1065" alt="image" src="https://github.com/user-attachments/assets/a10287c1-055c-4148-a013-9782748f37f0" />


---

## 🟢 1. Descriptive Statistics

**Definition**:  
Descriptive statistics are methods used to **summarize or describe** the main features of a dataset.

**Examples**:
- Average height of students in a class.
- Most common blood group in a survey.
- Spread of marks in an exam using standard deviation.

---

## 🟢 2. Inferential Statistics

**Definition**:  
Inferential statistics involve using **a sample to make conclusions or predictions** about a larger population.

**Examples**:
- Estimating the percentage of people in a city who prefer electric cars using a survey of 100 people.
- Testing if a new teaching method improves student performance using a sample class.

---

## 🧩 Types Based on Number of Variables

### 🟠 3. Univariate Analysis

**Definition**:  
Analysis involving **only one variable**.

**Examples**:
- Finding the average salary of employees.
- Creating a bar chart of student counts in departments.

---

### 🟠 4. Bivariate Analysis

**Definition**:  
Analysis involving **two variables**, usually to find relationships.

**Examples**:
- Study time vs exam score.
- Correlation between height and weight.

---

### 🟠 5. Multivariate Analysis

**Definition**:  
Analysis involving **more than two variables** at the same time.

**Examples**:
- Predicting student performance using study hours, attendance, and past marks.
- Classifying customer types using age, income, and purchase frequency.

---

## 🔷 Bivariate Statistical Techniques

### 🔹 6. Correlation

**Definition**:  
Measures how **strongly two variables are related**.

**Examples**:
- Positive correlation: Height and weight.
- Negative correlation: TV hours and exam score.

---

### 🔹 7. Regression

**Definition**:  
Predicts the value of a dependent variable using one or more independent variables.

- **Simple Regression**: One predictor  
  _Example_: Predicting weight using height.

- **Multivariate Regression**: Multiple predictors  
  _Example_: Predicting house price using area, number of rooms, and location.

---

# Linear Discriminant Analysis (LDA)

**Linear Discriminant Analysis (LDA)** is a dimensionality reduction technique used as a **preprocessing step** in **pattern classification** and **machine learning applications**. 
It finds the **linear combination of features** that best separates two or more classes.

---

## Main Idea

- To **find projection to a line** such that samples from different classes are **well separated linearly**.
- Or to **project high-dimensional data** (N-dimensional) onto a **lower-dimensional space** (K = n−1) while retaining the **maximum class discriminatory information**.

---

## Why is LDA Important?

1. It is **multi-faceted** and can handle multiple and different scenarios.
2. It can be used as a **multi-class linear classifier**, unlike logistic regression.
3. Useful for **dimensionality reduction**.
4. Helps in **extracting features** in face detection and image classification models.

---

## How Does LDA Work?

The goal of an LDA algorithm is to find the **best linear combination** that gives the **maximum separation between the number of groups**.
 LDA maximizes the ratio of between-class scatter to within-class scatter.

- It calculates **discriminant scores** using a **linear combination** of weights and centered data points.
- These weights are **extracted from eigenvectors**.

---

## What are the Limitations of LDA?

- It assumes **linear boundaries** and does **not work well for non-linear problems**.
- Assumes **normal distribution** of features.
- Performs poorly with **imbalanced data**.

---

## How to Prepare Data for LDA?

1. **Classification Problem**: LDA is applied when the **target variable is categorical** (binary or multi-class).
2. **Gaussian Distribution**: The standardized input variables should follow a **Gaussian distribution**.
3. **Remove Outliers**: LDA is **sensitive to outliers**. They should be removed to avoid distorting class means and variances.
4. **Same Variance**: LDA assumes **equal variance** across groups.
5. **Standardization**: It is better to **standardize the data** before applying LDA.

---

## Extensions to LDA

1. **Quadratic Discriminant Analysis (QDA)**:  
   Allows each class to have its own estimate of variance.
    <img width="266" height="178" alt="image" src="https://github.com/user-attachments/assets/2db3d69d-5586-48c5-9c40-622d2d4fbf20" />

2. **Regularized Discriminant Analysis (RDA)**:  
   Adds regularization to manage small sample sizes and high-dimensional data.

3. **Flexible Discriminant Analysis (FDA)**:  
   Allows for **non-linear combinations** of input variables using splines or kernel functions.

---

## Real-World Applications of LDA

1. **Face Recognition**:  
   Face recognition systems represent each face as a set of features and use LDA to classify individuals.

2. **Medical Diagnosis**:  
   Used to classify patients based on symptoms into disease categories.

3. **Customer Identification**:  
   LDA helps in segmenting customers likely to buy a product in a shopping mall.

4. **Speech Recognition**:  
   Helps classify spoken phrases based on sound patterns.

5. **Robotics & AI**:  
   Used in training robots for object recognition, interaction, and movement planning.

---

# Statistics Associated with Discriminant Analysis

Discriminant Analysis involves several key statistical measures and outputs that help evaluate the model's accuracy, interpretability, and performance.

## 1. Discriminant Scores
The score assigned to each observation, computed as:

$$
\text{Score} = (b_1 \cdot x_1 + b_2 \cdot x_2 + \dots + b_n \cdot x_n + \text{constant})
$$

Where:
- $b_i$: unstandardized coefficient
- $x_i$: value of the i-th variable

The scores are used to **assign the observation to the closest group centroid**.

## 2. Discriminant Function Coefficients
These are the **weights or multipliers** applied to each predictor variable in the discriminant function.

They are in the **original units** of measurement.

## 3. Standardized Discriminant Function Coefficients
These are the discriminant function coefficients calculated using **standardized variables** (mean = 0, std dev = 1).

They allow us to **compare the relative importance** of each variable in the model.

---

## 4. Canonical Correlation
Canonical correlation measures the **strength of association** between the discriminant scores and the groups.

It is essentially the **multiple correlation** between the set of predictor variables and the discriminant function.


## 5. Centroid
The centroid is the **mean value of the discriminant scores** for a particular group.

It shows where each group lies along the discriminant axis and helps in **group separation and classification**.
<img width="416" height="405" alt="image" src="https://github.com/user-attachments/assets/7bd3dbfe-5670-4e0c-9dcf-c20d9326a1cf" />



## 6. Classification Matrix (Confusion Matrix)
Also known as the **confusion matrix**, it shows the number of:

- **Correctly classified cases**
- **Misclassified cases**

It helps in evaluating the **accuracy of the discriminant function**.
<img width="250" height="241" alt="image" src="https://github.com/user-attachments/assets/e1ba364e-fcb0-48b2-b089-05dd8ca873d8" />


## 7. Eigenvalues  
For each discriminant function, the **eigenvalue** represents the **ratio of between-group to within-group variance**.

A higher eigenvalue means better discrimination between groups.
<img width="1200" height="630" alt="image" src="https://github.com/user-attachments/assets/a51e79e9-42d1-4562-ac61-45c452d57897" />


## 7. Pooled Within-Group Correlation Matrix
This matrix is obtained by **averaging the covariance matrices** of all groups.

It is used in computing the discriminant functions assuming **equal variance** among groups.



## 9. Structure Correlations (Pooled Within-Groups Correlation)

These represent the **simple correlations** between each predictor variable and the discriminant function.

They help interpret **which variables contribute most** to group separation.

---

# Linear Discriminant Analysis (LDA) -  Main Objective
- LDA works by finding directions in the feature space that best separate the classes. 
- Maximize between-class variance while minimizing within-class variance.

---

## Algorithm Steps

Assume we have two classes with d-dimensional samples such as $x_1,x_2,x_3,...,x_n$ where $x_i$ represents a data point.
Suppose $n_1$ samples belong to class $c_1$ and $n_2$ samples belong to class $c_2$. 
Let the means of class $c_1$ and class $c_2$ is $\mu_1$ and $\mu_2$ respectively. 
​
 
#### 1. Compute Within-Class Scatter Matrix ($S_w$)

- For each class $C_i$ with mean $\mu_i$:
  $$S_i = \sum_{x \in C_i}(x - \mu_i)(x - \mu_i)^T$$
- Combine to get:
  $$S_W = \sum_i S_i= \sum S_1+ \sum S_2+...$$
  

#### 2. Compute Between-Class Scatter Matrix ($S_B$)

- For two classes:
  $$S_B = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T$$
- For multiple classes, generalize using weighted class means.

### 3. Solve the Eigenproblem
The goal is to maximize the ratio of the between-class scatter to the within-class scatter, which leads us to the following criteria:
$$J=\frac{S_B}{S_W}$$.
For the best separation, we calculate the eigenvector corresponding to the highest eigenvalue of the scatter matrices
- Solve:
  $$S_W^{-1} S_B V = \lambda V$$
- Principal eigenvector (largest $\lambda\$) gives the **optimal projection direction**.

### 4. Project and Classify

- New observation $x$ is projected as:
  $$y = V^T x$$
- Assign class based on **closest centroid** or highest discriminant score.

---
# Linear Discriminant Analysis: Worked-Out Example

This demonstrates the complete LDA computation using matrix algebra, including:

- Mean vectors
- Within-class scatter matrix $S_W$
- Between-class scatter matrix $S_B$
- Solving the eigenvalue equation
- Computing eigenvectors
- Final projection vector

---
Given two classes:

**Class $C_1$:**

$$C_1 = \{\{(4,1), (2,4), (2,3), (3,6), (4,4)\}\} =  
\begin{bmatrix}
4&1\\
2& 4\\
2&3\\
3&6\\
4&4
\end{bmatrix}$$

**Class $C_2$:**

$$C_2 = \{(9,10), (6,8), (9,5), (8,7), (10,8)\}= \begin{bmatrix}
9&10\\
6& 8\\
9&5\\
8&7\\
10&8
\end{bmatrix}$$

<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/3768d5bb-bb62-44c8-acad-3b288ae99d1d" />

---

## ✅ Step 1: Compute Class Means

Compute the mean for each class:

$\mu_1 = \left( \frac{4+2+2+3+4}{5}, \frac{1+4+3+6+4}{5} \right) = (3.0,\ 3.6)$

$\mu_2 = \left( \frac{9+6+9+8+10}{5}, \frac{10+8+5+7+8}{5} \right) = (8.4,\ 7.6)$

<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/0e2e28ac-f405-45f8-98e1-b163f65b8e1b" />

These are your cluster centroids — the "center" of each group.

---

## ✅ Step 2: Compute Within-Class Scatter Matrix $S_W$

Each term in:

$$
S_i = \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T
$$

### For $C_1$:

Compute $(x - \mu_1)$ for each $x$ in $C_1$. 

- For $(4,1)$: $x_1 - \mu_1 =(4,1)-(3,3.6)= (1, -2.6)$
- For $(2,4)$: $x_2 - \mu_1 =(2,4)-(3,3.6)= (-1, 0.4)$
- For $(2,3)$: $x_3 - \mu_1 =(2,3)-(3,3.6)= (-1, -0.6)$
- For $(3,6)$: $x_4 - \mu_1 =(3,6)-(3,3.6)= (0, 2.4)$
- For $(4,4)$: $x_5 - \mu_1 =(4,4)-(3,3.6)= (1, 0.4)$

<img width="489" height="385" alt="image" src="https://github.com/user-attachments/assets/d3b628c0-3fed-4132-b21b-db9f87f582c1" />

Then:

$$\begin{align*}
S_1 
&=
\frac{1}{n} \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T \\
&=
\frac{1}{5}( \begin{bmatrix}1 \\
-2.6 \end{bmatrix} 
 \begin{bmatrix}1 & -2.6 \end{bmatrix} + \begin{bmatrix}-1 \\
0.4 \end{bmatrix} 
 \begin{bmatrix}-1 & 0.4 \end{bmatrix} + \begin{bmatrix}-1 \\
-0.6 \end{bmatrix} 
 \begin{bmatrix}-1 & -0.6 \end{bmatrix}+ \begin{bmatrix} 0 \\
2.4 \end{bmatrix} 
 \begin{bmatrix}0 & 2.4 \end{bmatrix}+\begin{bmatrix}1 \\
0.4 \end{bmatrix} 
 \begin{bmatrix}1 & 0.4 \end{bmatrix}) \\
 &=
\frac{1}{5}( \begin{bmatrix}
1 & -2.6 \\
-2.6 & 6.76
\end{bmatrix} + \begin{bmatrix}
1 & -0.4 \\
-0.4 & 0.16
\end{bmatrix} + \begin{bmatrix}
1 & 0.6 \\
0.6 & 0.36
\end{bmatrix} + \begin{bmatrix}
0 & 0 \\
0 & 5.76
\end{bmatrix} + \begin{bmatrix}
1 & 0.4 \\
0.4 & 0.16
\end{bmatrix}) \\
&=
\frac{1}{5}(\begin{bmatrix}
4 & -2 \\
-2 & 13.2
\end{bmatrix}) \\
&=
\begin{bmatrix}
0.8 & -0.4 \\
-0.4 & 2.64
\end{bmatrix} \\
\end{align*}$$



### For $C_2$:

Compute $(x - \mu_2)$ for each $x$ in $C_2$. 

- For $(9,10)$: $x_1 - \mu_2 =(9,10)-(8.4,\ 7.6)= (0.6, 2.4)$
- For $(6,8)$: $x_2 - \mu_2 =(6,8)-(8.4,\ 7.6)= (-2.4, 0.4)$
- For $(9,5)$: $x_3 - \mu_2 =(9,5)-(8.4,\ 7.6)= (0.6, -2.6)$
- For $(8,7)$: $x_4 - \mu_2 =(8,7)-(8.4,\ 7.6)= (-0.4, -0.6)$
- For $(10,8)$: $x_5 - \mu_2 =(10,8)-(8.4,\ 7.6)= (1.6, 0.4)$
  
<img width="484" height="390" alt="image" src="https://github.com/user-attachments/assets/dbc8a616-96f1-4ea2-a548-d915cdc60015" />


Then:

$$\begin{align*}
S_1 
&=
\frac{1}{n} \sum_{x \in C_i} (x - \mu_i)(x - \mu_i)^T \\
&=
\frac{1}{5}( \begin{bmatrix}0.6 \\
2.4 \end{bmatrix} 
 \begin{bmatrix}0.6 & 2.4  \end{bmatrix} + \begin{bmatrix}-2.4 \\
0.4 \end{bmatrix} 
 \begin{bmatrix}-2.4 & 0.4 \end{bmatrix} + \begin{bmatrix}0.6 \\
-2.6 \end{bmatrix} 
 \begin{bmatrix}0.6 & -2.6 \end{bmatrix}+ \begin{bmatrix} -0.4 \\
 -0.6 \end{bmatrix} 
 \begin{bmatrix}-0.4 &  -0.6 \end{bmatrix}+\begin{bmatrix}1.6 \\
0.4 \end{bmatrix} 
 \begin{bmatrix}1.6 & 0.4 \end{bmatrix}) \\
 &=
\frac{1}{5}( \begin{bmatrix}
0.36 & 1.44 \\
1.44 & 5.76
\end{bmatrix} + \begin{bmatrix}
5.76 & -0.96 \\
-0.96 & 0.16
\end{bmatrix} + \begin{bmatrix}
0.36 & -1.56 \\
-1.56 & 6.76
\end{bmatrix} + \begin{bmatrix}
0.16 & 0.24 \\
0.24 & 0.36
\end{bmatrix} + \begin{bmatrix}
2.56 & 0.64 \\
0.64 & 0.16
\end{bmatrix}) \\
&=
\frac{1}{5}(\begin{bmatrix}
9.2 & -0.2 \\
-0.2 & 13.2
\end{bmatrix}) \\
&=
\begin{bmatrix}
1.84 & -0.04 \\
-0.04 & 2.64
\end{bmatrix} \\
\end{align*}$$


### Total Within-Class Scatter Matrix:

$$
S_W = S_1 + S_2 = \begin{bmatrix}
0.8 & -0.4 \\
-0.4 & 2.64
\end{bmatrix} + \begin{bmatrix}
1.84 & -0.04 \\
-0.04 & 2.64
\end{bmatrix} =
\begin{bmatrix}
2.64 & -0.44 \\
-0.44 & 5.28
\end{bmatrix}
$$

---

## ✅ Step 3: Compute Between-Class Scatter Matrix $S_B$

$$\mu_1 - \mu_2 = (3,3.6) - (8.4,\ 7.6)= (-5.4 ,-4.0)$$

<img width="461" height="392" alt="image" src="https://github.com/user-attachments/assets/77e60f3a-8667-4dd3-8885-83c9660bce61" />

Then:

$$
S_B = (\mu_1 - \mu_2)(\mu_1 - \mu_2)^T = \begin{bmatrix} -5.4 \\
-4 \end{bmatrix} 
 \begin{bmatrix} -5.4 & -4  \end{bmatrix} =
\begin{bmatrix}
29.16 & 21.6 \\
21.6 & 16
\end{bmatrix}
$$

---

## ✅ Step 4: Compute $S_W^{-1} S_B$

The goal is to maximize the ratio of the between-class scatter to the within-class scatter: So maximise $$J=\frac{S_B}{S_W} = S_W^{-1} S_B$$.

First, compute the inverse of $S_W$:

$$
S_W =
\begin{bmatrix}
2.64 & -0.44 \\
-0.44 & 5.28
\end{bmatrix}
$$

Let $S_W^{-1}$ be:

$$
S_W^{-1} = \frac{1}{\det(S_W)} \cdot adj(A) = 
\frac{1}{(2.64 \cdot 5.28) - (-0.44)^2} \cdot
\begin{bmatrix}
5.28 & 0.44 \\
0.44 & 2.64
\end{bmatrix} = \begin{bmatrix}
0.384 & 0.032 \\
0.032 & 0.192
\end{bmatrix}
$$

Then:

$$S_W^{-1} S_B =\begin{bmatrix}
0.384 & 0.032 \\
0.032 & 0.192
\end{bmatrix}
\cdot
\begin{bmatrix}
29.16 & 21.6 \\
21.6 & 16
\end{bmatrix}=
\begin{bmatrix}
11.89 & 8.81 \\
5.08 & 3.76
\end{bmatrix}
$$

---

## ✅ Step 5: Find Eigenvalues

Solve:

$$
\left| S_W^{-1}S_B - \lambda I \right| = 0
$$

$$
\left|
\begin{bmatrix}
11.89 & 8.81 \\
5.08 & 3.76
\end{bmatrix} -  \lambda  \cdot \begin{bmatrix}1 & 0 \\
0 & 1
\end{bmatrix}
\right| = 0
$$

$$
\left|
\begin{bmatrix}
11.89 - \lambda & 8.81 \\
5.08 & 3.76 - \lambda
\end{bmatrix}
\right| = 0
$$

Determinant:

$$
(11.89 - \lambda)(3.76 - \lambda) - (8.81)(5.08) = 0
$$

Solve:

$$ 44.7064 - 11.89\lambda- 3.76\lambda + \lambda^2 -44.7548 = 0$$

$$
\lambda^2 - 15.65\lambda + 0.0484 = 0
$$

$$
\Rightarrow \lambda = \frac{15.65 \pm \sqrt{(-15.65)^2 - 4 \cdot 1 \cdot  0.0484}}{2 \cdot 1} = 15.65, 0.0031
$$

This is the number that tells how much separation or discrimination power the direction (eigenvector) gives.

---

## ✅ Step 6: Find Eigenvector $v$

For the best separation, we calculate the eigenvector corresponding to the highest eigenvalue of the scatter matrices.

So $\lambda =15.65$

Now Solve:

$$
(S_W^{-1} S_B - \lambda I)v = 0
$$

Substitute and solve:
$$
\begin{bmatrix}
11.89 - 15.65 & 8.81 \\
5.08 & 3.76 - 15.65
\end{bmatrix}
\cdot
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
= 0
$$

$$
\begin{bmatrix}
-3.76 & 8.81 \\
5.08 & -11.89
\end{bmatrix}
\cdot
\begin{bmatrix}
v_1 \\
v_2
\end{bmatrix}
= 0
$$

From row 1: $-3.76v_1 + 8.81v_2 = 0$

$$
\Rightarrow \frac{v_1}{v_2} = \frac{8.81}{3.76} = 2.34
$$

So:

$$
v =
\begin{bmatrix}
2.34 \\
1
\end{bmatrix}
$$

Normalize:

$$\|v\| = \sqrt{(2.34)^2 + 1^2} = 2.54
\Rightarrow v =
\begin{bmatrix}
\frac{2.34}{2.54} \\
\frac{1}{2.54}
\end{bmatrix}=
\begin{bmatrix}
0.92 \\
0.39
\end{bmatrix}
$$

This is the direction that maximally separates the two clusters (centroids).
<img width="580" height="519" alt="image" src="https://github.com/user-attachments/assets/614b647c-97a4-4833-96a4-cb4ded9f3eab" />

---

## ✅ Step 7: Project Data

Each $x$ is projected onto new axis:

$$
\text{Discriminant Score} =y = v^T x = [0.92,\ 0.39] \cdot x = 0.92 \cdot x_1 + 0.39 \cdot x_2
$$


​To classify or visualize, we **project each data point onto the eigenvector** and find the **discriminant score**

For example,
- For a point in \$C\_1\$, say \$(4,1)\$:

$$
y = 0.92 \cdot 4 + 0.39 \cdot 1 = 3.68 + 0.39 = 4.07
$$

- For a point in \$C\_2\$, say \$(9,10)\$:

$$
y = 0.92 \cdot 9 + 0.39 \cdot 10 = 8.28 + 3.9 = 12.18
$$

---

## 📌 Summary of Results

| Step | Description |
|------|-------------|
| Step 1 | Class Means |
| Step 2 | Within-Class Scatter Matrix $S_W$ |
| Step 3 | Between-Class Scatter Matrix $S_B$ |
| Step 4 | Matrix Product $S_W^{-1} S_B$ |
| Step 5 | Find $\lambda$ |
| Step 6 | Solve Eigenvector $v$ |
| Step 7 | Project $x$ onto new axis using $y = v^T x$ |

---

## 📖 References

- LDA Theory: [Wikipedia](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
- Python Tutorial: [GeeksforGeeks LDA](https://www.geeksforgeeks.org/machine-learning/ml-linear-discriminant-analysis/)
