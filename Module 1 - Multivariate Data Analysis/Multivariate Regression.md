# Regression Models and Conditional Distributions

Regression models are powerful tools used to understand and predict the relationship between a **dependent variable** (often $Y$) and one or more **independent variables** (often $X$). These models predict the value of the dependent variable based on the values of the independent variables. 

---

## What Is a Regression Model?

A **regression model** attempts to estimate:

$$
\mathbb{E}[Y \mid X] = f(X) = \beta_0 + \beta_1 X
$$

Where:
- $Y$ is the dependent or output variable
- $X$ is the independent or input variable(s)
- $f(X)$ is a function, often assumed to be linear in basic models



And it assumes the **conditional distribution** of $Y$ given $X$ is **normal**:

$$
Y \mid X = x \sim \mathcal{N}(\beta_0 + \beta_1 x, \sigma^2)
$$

That is, for a fixed value of $X$, the values of $Y$ follow a **normal distribution** centered at the regression line.

---

## Example: Predicting House Prices

Let’s say you're predicting house prices based on square footage.

- Let $X$ = square footage  
- Let $Y$ = house price

The model assumes:

$$
Y \mid X = x \sim \mathcal{N}(\beta_0 + \beta_1 x, \sigma^2)
$$

So for houses with 1500 sqft, prices are **normally distributed** around a **mean predicted by the regression line**.

---

## Conditional Distribution in Bivariate Normal

From the bivariate normal theory, we know:

If

$$
(X, Y) \sim BVN\left(
\begin{bmatrix} \mu_X \\ \mu_Y \end{bmatrix},
\begin{bmatrix}
\sigma_X^2 & \rho \sigma_X \sigma_Y \\
\rho \sigma_X \sigma_Y & \sigma_Y^2
\end{bmatrix}
\right)
$$

**The conditional mean formula:**

$$
\mu_{Y \mid X = x} =\mathbb{E}[Y \mid X = x] = \mu_Y + \rho \frac{\sigma_Y}{\sigma_X} (x - \mu_X)
$$

Can be rearranged as:

$$
\mu_{Y \mid X = x}= 
\left( \mu_Y - \rho \frac{\sigma_Y}{\sigma_X} \mu_X \right)
+ 
(\rho \frac{\sigma_Y}{\sigma_X}) x = \beta_0 + \beta_1 x
$$

This is the **regression line** derived directly from the **conditional mean**.

So regression is naturally embedded in the bivariate normal distribution.

---



# Assumptions of Multivariate Regression
---

###  1. **Linearity**

**What it means**:
There should be a straight-line relationship between the inputs (independent variables) and the result (dependent variable).

**What if it’s not true?**
The model will give wrong results because it's trying to fit a straight line to a curved pattern.

**How to check**:

* **Residuals vs Fitted Plot**: If there's a curve, it's not linear.
* **Partial Residual Plot**: Helps show how each variable affects the result.

**Example**:
Predicting salary using years of experience is linear. But predicting salary using age (very young and very old may earn less) may not be linear.

---

### 2. **Independence of Observations**

 **What it means**:
Each observation should be unrelated to the others.

 **What if it’s not true?**
Standard errors become too small, making unimportant predictors look important.

 **How to check**:

* **Durbin-Watson Test**: Detects patterns in data over time.
* **Residual Plot**: Patterns mean values are connected.

 **Example**:
In sales data collected every day, today’s sales may depend on yesterday’s (violates independence).

---

### 3. **Homoscedasticity (Equal Spread of Errors)**

 **What it means**:
Errors (residuals) should have the same spread no matter the value of the inputs.

 **What if it’s not true?**
The model might give more weight to certain data points, leading to incorrect conclusions.

 **How to check**:

* **Residual vs Fitted Plot**: Should be a random cloud. A cone or fan shape indicates a problem.
* **Scale-Location Plot**: Checks if the spread is constant.

 **Example**:
Predicting house prices—if errors are large only for expensive houses, it's a violation.

---

### 4. **Normality of Errors**

 **What it means**:
The errors (residuals) should follow a normal (bell-curve) distribution.

 **What if it’s not true?**
The confidence intervals and p-values might be wrong.

 **How to check**:

* **Q-Q Plot**: Dots should lie on a straight line.
* **Histogram of Residuals**: Should look like a bell curve.

 **Example**:
If your errors are skewed, like only large positive errors, it's not normal.

---

### 5. **No Multicollinearity**

 **What it means**:
The input variables (predictors) shouldn’t be highly related to each other.

 **What if it’s not true?**
It’s hard to know which variable is actually influencing the output, and the model becomes unstable.

 **How to check**:

* **VIF (Variance Inflation Factor)**: Value > 10 is a red flag.
* **Correlation Matrix**: Look for high correlations between inputs.

 **Example**:
Including both height in inches and height in centimeters will cause multicollinearity.

---

### 6. **No Autocorrelation in Residuals**

 **What it means**:
The errors should not follow a pattern over time (or space).

 **What if it’s not true?**
Future predictions, especially in time-series data, will be unreliable.

 **How to check**:

* **Durbin-Watson Test**
* **Residuals Over Time Plot**

 **Example**:
In stock market data, prices today are linked to yesterday’s prices – autocorrelation is common.

---

### 7. **Sufficient Sample Size**

 **What it means**:
You need enough data to build a reliable model.

 **What if it’s not true?**
The model might learn noise (random things) instead of the real pattern—called overfitting.

 **How to check**:

* **Power Analysis**: To estimate needed sample size.

 **Example**:
Trying to predict house prices using only 10 houses is too little—model won’t generalize well.

---

###  8. **Mean of Residuals Should Be Zero**

 **What it means**:
On average, the prediction errors (residuals) should cancel out to zero.

 **What if it’s not true?**
The line may not be the best fit—it's either always over or under-predicting.

 **How to check**:

* Look at the **mean of residuals** after fitting the model.

 **Example**:
If you predict that all students will get 5 marks more than they actually do, the residuals are not centered around 0.


---

## ✅ **Summary - The key assumptions underlying a multivariate regression model**


| Assumption                                               | Description                                                                                   |
| -------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **1. Linearity**                                         | The relationship between independent and dependent variables is linear.                       |
| **2. Independence of observations**                      | Each observation (data point) is independent of others.                                       |
| **3. Multivariate normality**                            | The dependent variables (DVs) are **jointly normally distributed** for each level of the IVs. |
| **4. Homoscedasticity**                                  | The **variance of the residuals** is constant across all levels of the independent variables. |
| **5. No multicollinearity**                              | Independent variables are not highly correlated with each other.                              |
| **6. Covariance matrices equality (for MANOVA/MANCOVA)** | Groups should have equal **variance-covariance matrices**.                                    |

---

### 📘 Example:

A model predicting both **math** and **science scores** from **hours studied** must assume:

* Both scores are **normally distributed** for given hours
* The effect of hours is **linear**
* Errors have **equal variance**

---

### 🚫 Effect of Assumption Violations:

| Violation                       | Impact on Model                                                                | 
| ------------------------------- | ------------------------------------------------------------------------------ |
| **Linearity**                   | Predictors may have **nonlinear effects**, leading to bias in predictions      |
| **Independence**                | Causes **underestimation of standard errors**, inflating type I error          | 
| **Non-normality**               | Affects **p-values** and **confidence intervals**, especially in small samples | 
| **Heteroscedasticity**          | Leads to **inefficient estimates** and biased standard errors                  | 
| **Multicollinearity**           | Regression coefficients become **unstable** and hard to interpret              | 
| **Unequal covariance matrices** | Results in **incorrect significance testing** in MANOVA/MANCOVA                |                                                     

---

## 📊 Question: Multivariate Linear Regression

The data in the table relate grams of plant dry weight $Y$, to present soil organic matter $X_1$ and kilograms of supplemental soil nitrogen added per 100 square meters $X_2$.

| Y (Dry weight) | X₁ (Soil organic matter) | X₂ (Soil nitrogen) |
|----------------|---------------------------|---------------------|
| 78.5           | 7                         | 2.6                 |
| 74.3           | 1                         | 2.9                 |
| 104.3          | 11                        | 5.6                 |
| 87.6           | 11                        | 3.1                 |

**i.** Obtain the multivariate regression equation.

**ii.** Predict the dry weight when soil organic matter = 5 and soil nitrogen = 4.



## ✅ Answer

We want to fit a multivariate linear regression model of the form:

$$
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2
$$



### Step 1: Set up the matrices

Let:

$$
X = 
\begin{bmatrix}
1 & 7 & 2.6 \\
1 & 1 & 2.9 \\
1 & 11 & 5.6 \\
1 & 11 & 3.1 \\
\end{bmatrix},
\quad
Y = 
\begin{bmatrix}
78.5 \\
74.3 \\
104.3 \\
87.6 \\
\end{bmatrix}
$$


### Step 2: Use the Normal Equation

We compute:

$$
\boldsymbol{\beta} = (X^T X)^{-1} X^T Y
$$


#### Compute $X^T X$

$$
X^T X =
\begin{bmatrix}
1 & 1 & 1 & 1 \\
7 & 1 & 11 & 11 \\
2.6 & 2.9 & 5.6 & 3.1 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
1 & 7 & 2.6 \\
1 & 1 & 2.9 \\
1 & 11 & 5.6 \\
1 & 11 & 3.1 \\
\end{bmatrix} =
\begin{bmatrix}
4 & 30 & 14.2 \\
30 & 292 & 116.8 \\
14.2 & 116.8 & 56.14 \\
\end{bmatrix}
$$

Then 

$$
(X^T X)^{-1} = 
\begin{bmatrix}
4 & 30 & 14.2 \\
30 & 292 & 116.8 \\
14.2 & 116.8 & 56.14 \\
\end{bmatrix}^{-1} = 
\begin{bmatrix}
2.4752 & -0.0230725 & -0.578072 \\
-0.0230725 & 0.0206249 & -0.0370744 \\
-0.578072 & -0.0370744 & 0.241163 \\
\end{bmatrix}
$$


#### Compute $X^T Y$

$$
X^T Y =
\begin{bmatrix}
1 & 1 & 1 & 1 \\
7 & 1 & 11 & 11 \\
2.6 & 2.9 & 5.6 & 3.1 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
78.5 \\
74.3 \\
104.3 \\
87.6 \\
\end{bmatrix} =
\begin{bmatrix}
344.7 \\
2734.7 \\
1275.21 \\
\end{bmatrix}
$$


#### Compute $\boldsymbol{\beta} = (X^T X)^{-1} X^T Y$


$$
\boldsymbol{\beta} = (X^T X)^{-1} X^T Y
= \begin{bmatrix}
2.4752 & -0.0230725 & -0.578072 \\
-0.0230725 & 0.0206249 & -0.0370744 \\
-0.578072 & -0.0370744 & 0.241163 \\
\end{bmatrix} 
\cdot
\begin{bmatrix}
344.7 \\
2734.7 \\
1275.21 \\
\end{bmatrix}
= \begin{bmatrix}
52.9416 \\
1.17213 \\
6.88518 \\
\end{bmatrix}
$$



### ✅ Final Regression Equation:

$$
\hat{Y} =
$$







### Step 3: Prediction

Let $X_1 = 5$, $X_2 = 4$. Then:

$$
\hat{Y} = 
$$



### 📌 Final Answer:

- **Regression equation:**  
  $$\hat{Y} = 70.06 + 2.75 X_1 + 0.62 X_2$$

- **Prediction at $X_1 = 5$, $X_2 = 4$:**  
  $$\hat{Y} = 86.29$$


  ---


  # Multivariate Analysis of Variance (MANOVA)

**MANOVA (Multivariate Analysis of Variance)** compares the means of **multiple dependent variables** across groups defined by one or more **independent variables**.

Unlike ANOVA (which tests group differences for a single dependent variable), **MANOVA** accounts for **correlations between multiple dependent variables**, offering a more **comprehensive understanding** of group differences.

---

## Key Assumptions of MANOVA

- **Multivariate Normality**:  
  All dependent variables should be normally distributed **within each group**.

- **Homogeneity of Covariance Matrices**:  
  The variance-covariance matrices of the dependent variables should be **equal across groups**.

- **Independence of Observations**:  
  Observations should be independent — one participant's data must not influence another's.

- **Linearity**:  
  The relationships between each pair of dependent variables should be **linear within each group**.

---

## Why Assumption Checking Matters

| Reason                  | Explanation                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| Validity of Results   | Violating assumptions can lead to **misleading or invalid** conclusions.    |
| Interpretation        | Proper interpretation relies on assumption validity. If assumptions fail, consider alternative methods. |
| Statistical Power     | Satisfying assumptions improves the ability to detect **true** group differences. |

---

## Types of MANOVA

| Type          | Description                                                  |
|---------------|--------------------------------------------------------------|
| **One-way MANOVA** | Tests differences in multiple DVs based on a **single** IV.   |
| **Two-way MANOVA** | Examines the combined effects of **two** IVs on multiple DVs. |

---

## Example: Testing Diet Plans

### Objective:
Determine if different diet plans lead to differences in **health outcomes** like weight, blood pressure, and cholesterol.

### Hypotheses:
- **Null Hypothesis**: No significant difference in the combined health outcomes across diet plans.
- **Alternative Hypothesis**: At least one diet leads to a different combination of outcomes.

### Steps:
1. Check MANOVA assumptions.
2. Conduct MANOVA test.
3. If significant, run **post-hoc tests** to identify where differences lie.

---

## Benefits of Using MANOVA

- **Comprehensive Analysis**:  
  Understand group differences across multiple outcomes simultaneously.

- **Reduced Type I Error**:  
  Considers correlations between DVs, avoiding inflated error from running separate ANOVAs.

- **Increased Statistical Power**:  
  Higher likelihood of detecting actual group effects when DVs are correlated.

- **Informative Visualizations**:  
  Group differences and relationships can be visualized using scatter plots, profile plots, and canonical discriminant analysis.

---

## Practical Applications of MANOVA

| Domain                 | Use Case Example                                                                 |
|------------------------|----------------------------------------------------------------------------------|
| **Education**        | Compare teaching methods on **grades**, **participation**, and **motivation**.   |
| **Healthcare**       | Evaluate treatment effects on **weight**, **BP**, and **glucose**.               |
| **Marketing**        | Compare ad campaigns based on **recall**, **engagement**, and **conversion**.    |
| **Psychology**       | Assess therapy outcomes on **anxiety**, **mood**, and **stress**.                |

---

# Multivariate Analysis of Covariance (MANCOVA)

**MANCOVA** is a statistical technique used to evaluate group differences across multiple dependent variables **while controlling for covariates**. It provides deeper insights into relationships among variables and helps reduce error by adjusting for known influences.

---

## Overview of MANCOVA

- **MANCOVA** = MANOVA + Covariates
- Analyzes multiple dependent variables while **controlling for continuous covariates**
- Helps clarify the **true effect** of the independent variable(s) on the dependent variables
- Leads to more **accurate and insightful conclusions**

---

## Assumptions of MANCOVA

To use MANCOVA correctly, the following assumptions must be satisfied:

- **Multivariate Normality**:  
  All dependent variables should follow a multivariate normal distribution within each group

- **Homogeneity of Variance-Covariance Matrices**:  
  The variance-covariance matrices should be equal across all groups

- **Linearity**:  
  The relationships between dependent variables and covariates must be linear

- **Independence of Observations**:  
  Each data point must be independent of others

---

## Unique Assumptions of MANCOVA

In addition to the standard MANOVA assumptions, MANCOVA adds:

- **Homogeneity of Regression Slopes**:  
  The effect of covariates on dependent variables should be consistent across all groups

- **Measurement Level**:  
  Covariates should be **continuous**; categorical covariates require different techniques or encoding

---

## Role of Covariates in MANCOVA

| Benefit                    | Explanation                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| Reduce Error Variance   | By accounting for external influences (e.g., age, baseline scores)          |
| Adjust Group Means      | Clarifies the **true effect** of independent variables by removing covariate bias |
| Increase Statistical Power | Increases the precision and sensitivity of tests                         |

---

## MANCOVA in Action

**Research Question:**  
> Do different diet plans affect weight loss, cholesterol levels, and blood pressure?

### Variables:
- **Independent Variable (IV)**: Diet plan (A, B, C)
- **Dependent Variables (DVs)**: Weight loss, cholesterol, blood pressure
- **Covariates**: Initial weight, age, exercise level

### Steps:
1. Collect data
2. Check assumptions
3. Perform MANCOVA
4. Conduct post-hoc tests (if significant)
5. Interpret multivariate and univariate results

---

## Benefits of MANCOVA

| Benefit                      | Description                                                               |
|------------------------------|---------------------------------------------------------------------------|
| Reduced Error Variance     | Improves accuracy of effect estimates                                      |
| Increased Power            | Enhances ability to detect significant group differences                   |
| Adjusted Group Means       | Offers clearer insights into the true impact of the IVs                    |

---

## Interpreting MANCOVA Results

| Output Type         | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| **Multivariate Tests** | Assess group differences across the combined DVs (after controlling for covariates) |
| **Univariate Tests**   | Examine group differences for each DV individually (if multivariate test is significant) |
| **Effect Size**        | Important to determine practical significance, not just statistical       |

---

## Limitations and Considerations

- **Assumption Violations**  
  Compromise result validity — alternative methods may be required

- **Covariate Selection**  
  Must be thoughtful — irrelevant or missing covariates distort findings

- **Interpretation Complexity**  
  Relationships among variables can be complex and require care

- **Generalizability**  
  Results may be limited to the sample and context used in the study

---
