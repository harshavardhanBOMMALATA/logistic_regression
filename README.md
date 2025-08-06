# logistic_regression
---

This project offers a thorough exploration of **Logistic Regression**, guiding learners from beginner to advanced levels through clear explanations and practical examples. The primary objective is to not only understand the theoretical foundations of logistic regression but also to apply it to real-world problems. Through hands-on experience, learners will manually implement logistic regression using **Python**, without relying on high-level machine learning libraries, enabling a deep understanding of the underlying math and logic. The project covers both **binary classification** and **multiclass classification**, providing insight into how logistic regression can be used for tasks like spam detection, disease prediction, and customer churn analysis. Throughout the tutorial, real-world use cases are explored, and key concepts such as **model evaluation**, **cost functions**, and **optimization** are discussed in detail. Visualizations, including decision boundaries and graphical representations of model performance, will help learners see how logistic regression classifies data and makes predictions. By the end of this project, participants will have built a complete logistic regression model from scratch, gained an understanding of its practical applications, and developed the confidence to move on to more advanced machine learning algorithms.

---

### 🛠️ Tools & Technologies Used

* **📘 Jupyter Notebook** – for writing and executing the code step-by-step with explanations
* **🐍 Python** – core programming language used for building the logic
* **📂 CSV Files** – dataset used for training and predictions
* **📊 pandas** – for reading and handling the dataset
* **📈 matplotlib** – for visualizing the data and regression line
* **📐 numpy** – for performing numerical and statistical operations 

---

## 📁 File Structure 

1. **Introduction to Logistic Regression**
2. **Working and Why It Works**
3. **Mathematical Intuition**
4. **Implementation Without Scikit-Learn**
5. **Implementation With Scikit-Learn**
6. **Applications**
7. **Advantages and Disadvantages**

---

## 📘 Introduction to Logistic Regression

**Definition**:
Logistic Regression is a **supervised machine learning algorithm** used primarily for **classification tasks**. While traditional regression gives outputs ranging from **−∞ to +∞**, logistic regression applies the **sigmoid function** to map these values into the range **0 to 1**, making it suitable for predicting probabilities.

---

### 💡 Why "Regression" in Classification?

Even though the goal is classification, the algorithm initially computes a **continuous value** using a linear equation:

$$
z = w_1x_1 + w_2x_2 + \ldots + b
$$

This output `z` is then passed through the **sigmoid function** to squash it into a probability between 0 and 1:

$$
\hat{y} = \frac{1}{1 + e^{-z}}
$$

---

### 📚 What is Supervised Learning?

Logistic regression is a **supervised learning model**, meaning it learns from **labeled data** — where both input features and correct output labels are already known.

> 🧠 Example:
> A dataset containing features like `study_hours`, `sleep_hours`, and the label `verdict` (Pass or Fail). The model learns from this structured data to predict outcomes on unseen inputs.

---

### 🔄 Regression vs. Logistic Regression

* **Linear Regression** is used for predicting **continuous values** (e.g., price, temperature).
* **Logistic Regression** is used for **classification** by applying a regression-like linear combination and then using **sigmoid** to output probabilities.

---

### 🧭 Types of Logistic Regression

1. **Binary Classification** – Two output classes (e.g., Pass/Fail, Yes/No)
2. **Multiclass Classification** – More than two output classes (e.g., Low/Medium/High)

We’ll explore both types in the following sections.

---

## ⚙️ Working of Logistic Regression — and Why It Works

In most machine learning models — including logistic regression — the process involves **training first**, then **testing**, and finally **predicting**. During training, there are **3 core steps**:

---

### 🔹 Step 1: Prepare and Check the Dataset

Start with labeled data — meaning we know both the features and the correct outputs (like "Pass" or "Fail"). This helps the model learn patterns between input and output.

---

### 🔹 Step 2: Compute the Linear Function $z$

We use a **linear combination of weights and inputs**:

$$
z = w_1x_1 + w_2x_2 + \ldots + w_nx_n + b
$$

This is written as:

$$
z = \mathbf{w}^T \cdot \mathbf{x}
$$

The result of this equation is usually a **very large or small value**, possibly outside the range \[0, 1].

---

### 🔹 Step 3: Apply the Sigmoid Function

To **compress these large values between 0 and 1**, we apply the **sigmoid function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

This allows us to interpret the output as a **probability**. You can relate it to how logarithms are used to reduce large values in mathematics — sigmoid does something similar in ML.

---

### 🎯 Making Predictions

We usually apply a **threshold** (commonly 0.5):

* If $\hat{y} < 0.5$ → Predict **Class 0**
* If $\hat{y} \geq 0.5$ → Predict **Class 1**

This is how the model converts a probability into a final class prediction.

---

### 🎒 But What Are Weights?

**Weights** represent the **importance** of each input feature.

Let’s understand this with a simple example:

> Imagine you're a bank manager. A person comes and says:
> “Please give me a loan. Here's my name and my salary.”
>
> Now — does their **name** help you decide anything? Probably not.
> But their **salary**? Definitely — higher salary usually means higher loan eligibility.
>
> So in this dataset:
>
> * **Salary** has high weight (important feature)
> * **Name** has low or zero weight (not useful)

---

### ❓ Why Not Just Use a Straight Line?

A **linear regression model** (straight line) outputs values from **−∞ to +∞** — which doesn't make sense when you're trying to **classify into categories like 0 or 1**.

For example:

* A person might get predicted value of **2.3** or **−1.7**, but how do we interpret that as a class label?

Also:

* It doesn't map well to **probabilities**, and you can’t define a clean threshold on infinite-range output.

🛠️ That’s why we apply the **sigmoid function** to convert the raw linear output into a **probability between 0 and 1**, making classification **accurate, bounded, and interpretable**.

---

## 📐 Mathematical Intuition

To understand how logistic regression works internally, let’s take a simple dataset and walk through the math involved.

### 📊 Dataset Used:

| ID | Study Hours (x₁) | Sleep Hours (x₂) | Result (y) |
| -- | ---------------- | ---------------- | ---------- |
| 1  | 2                | 7                | 0          |
| 2  | 3                | 6                | 0          |
| 3  | 4                | 5                | 0          |
| 4  | 6                | 4                | 1          |
| 5  | 7                | 3                | 1          |
| 6  | 8                | 2                | 1          |

---

### 🧠 Why Start with Weights and Bias as Zero?

We begin with initial values:

* `w₁ = 0`, `w₂ = 0`, `b = 0`

This is a common practice in logistic regression to start from **neutral assumptions** — meaning:

* Every feature (like `Study Hours`, `Sleep Hours`) is initially given **no priority or influence** on the prediction.

* This reflects the idea:

  > *“If I know nothing about which feature matters, I start from zero and let the data guide me.”*

* **Bias `b = 0`** assumes that even without any input features, the prediction is neutral (sigmoid(0) = 0.5).

As we go through training (like Epoch 1), we **update weights and bias** based on how wrong the model was. Over time, features with more influence will gain higher weights, and those that don't matter will remain close to zero.

This is how logistic regression **learns priorities** from scratch.

---

### 🧮 Epoch 1 (Initial Weights: w₁ = 0, w₂ = 0, b = 0)

| ID | x₁ (Study) | x₂ (Sleep) | y | z     | ŷ = sigmoid(z) | y - ŷ   | (y - ŷ)·x₁ | (y - ŷ)·x₂ |
| -- | ---------- | ---------- | - | ----- | -------------- | ------- | ---------- | ---------- |
| 1  | 2          | 7          | 0 | 0.000 | 0.5000         | -0.5000 | -1.0000    | -3.5000    |
| 2  | 3          | 6          | 0 | 0.000 | 0.5000         | -0.5000 | -1.5000    | -3.0000    |
| 3  | 4          | 5          | 0 | 0.000 | 0.5000         | -0.5000 | -2.0000    | -2.5000    |
| 4  | 6          | 4          | 1 | 0.000 | 0.5000         | 0.5000  | 3.0000     | 2.0000     |
| 5  | 7          | 3          | 1 | 0.000 | 0.5000         | 0.5000  | 3.5000     | 1.5000     |
| 6  | 8          | 2          | 1 | 0.000 | 0.5000         | 0.5000  | 4.0000     | 1.0000     |

---

## 🧮 Epoch 1 — Gradient Summation

| Gradient Term     | Sum (Σ)     |
| ----------------- | ----------- |
| Σ(y - ŷ)·x₁ → Δw₁ | **6.0000**  |
| Σ(y - ŷ)·x₂ → Δw₂ | **-4.5000** |
| Σ(y - ŷ)   → Δb   | **0.0000**  |
---

### 📌 Updated Weights After Epoch 1:

$$
\begin{align*}
w₁ &= 0 + 0.01 × 6.000 = \textbf{0.0600} \\
w₂ &= 0 + 0.01 × (-4.500) = \textbf{-0.0450} \\
b  &= 0 + 0.01 × 0 = \textbf{0.0000}
\end{align*}
$$

---

## 🔁 **Epoch 2 Table**

| ID | x₁ (Study) | x₂ (Sleep) | y | z       | ŷ = sigmoid(z) | y - ŷ   | (y - ŷ)·x₁ | (y - ŷ)·x₂ |
| -- | ---------- | ---------- | - | ------- | -------------- | ------- | ---------- | ---------- |
| 1  | 2          | 7          | 0 | -0.1350 | 0.4663         | -0.4663 | -0.9326    | -3.2641    |
| 2  | 3          | 6          | 0 | 0.0000  | 0.5000         | -0.5000 | -1.5000    | -3.0000    |
| 3  | 4          | 5          | 0 | 0.1350  | 0.5337         | -0.5337 | -2.1348    | -2.6684    |
| 4  | 6          | 4          | 1 | 0.2100  | 0.5523         | 0.4477  | 2.6863     | 1.7908     |
| 5  | 7          | 3          | 1 | 0.2550  | 0.5634         | 0.4366  | 3.0562     | 1.3097     |
| 6  | 8          | 2          | 1 | 0.3000  | 0.5744         | 0.4256  | 3.4048     | 0.8512     |

---

## 🔢 **Delta Sums (Gradients)**

| Term        | Σ Value     |
| ----------- | ----------- |
| Σ(y - ŷ)·x₁ | **4.5799**  |
| Σ(y - ŷ)·x₂ | **-4.9809** |
| Σ(y - ŷ)    | **-0.1900** |

---

## 📌 **Updated Weights (after Epoch 2)**

Using learning rate `η = 0.01`:

$$
\begin{align*}
w₁ &= 0.0600 + 0.01 × 4.5799 = \textbf{0.1058} \\
w₂ &= -0.0450 + 0.01 × (-4.9809) = \textbf{-0.0948} \\
b  &= 0.0000 + 0.01 × (-0.1900) = \textbf{-0.0019}
\end{align*}
$$

---

## 🔁 **Epoch 3 Table**

| ID | x₁ | x₂ | y | z       | ŷ      | y - ŷ   | (y - ŷ)·x₁ | (y - ŷ)·x₂ |
| -- | -- | -- | - | ------- | ------ | ------- | ---------- | ---------- |
| 1  | 2  | 7  | 0 | -0.4814 | 0.3815 | -0.3815 | -0.7630    | -2.6707    |
| 2  | 3  | 6  | 0 | -0.3642 | 0.4101 | -0.4101 | -1.2303    | -2.4606    |
| 3  | 4  | 5  | 0 | -0.2470 | 0.4386 | -0.4386 | -1.7544    | -2.1932    |
| 4  | 6  | 4  | 1 | 0.2060  | 0.5513 | 0.4487  | 2.6920     | 1.7948     |
| 5  | 7  | 3  | 1 | 0.3796  | 0.5937 | 0.4063  | 2.8441     | 1.2189     |
| 6  | 8  | 2  | 1 | 0.5532  | 0.6350 | 0.3650  | 2.9199     | 0.7299     |

### 🔢 Gradient Sums

| Term        | Σ Value     |
| ----------- | ----------- |
| Σ(y - ŷ)·x₁ | **4.7083**  |
| Σ(y - ŷ)·x₂ | **-3.5809** |
| Σ(y - ŷ)    | **-0.0102** |

### 📌 **Updated Weights After Epoch 3**

$$
\begin{align*}
w₁ &= 0.1058 + 0.01 × 4.7083 = \textbf{0.1529} \\
w₂ &= -0.0948 + 0.01 × (-3.5809) = \textbf{-0.1306} \\
b  &= -0.0019 + 0.01 × (-0.0102) = \textbf{-0.0020}
\end{align*}
$$

---

## 🔁 **Epoch 4 Table**

| ID | x₁ | x₂ | y | z       | ŷ      | y - ŷ   | (y - ŷ)·x₁ | (y - ŷ)·x₂ |
| -- | -- | -- | - | ------- | ------ | ------- | ---------- | ---------- |
| 1  | 2  | 7  | 0 | -0.8037 | 0.3092 | -0.3092 | -0.6184    | -2.1644    |
| 2  | 3  | 6  | 0 | -0.6265 | 0.3483 | -0.3483 | -1.0449    | -2.0900    |
| 3  | 4  | 5  | 0 | -0.4492 | 0.3896 | -0.3896 | -1.5584    | -1.9479    |
| 4  | 6  | 4  | 1 | 0.3215  | 0.5796 | 0.4204  | 2.5222     | 1.6817     |
| 5  | 7  | 3  | 1 | 0.5615  | 0.6368 | 0.3632  | 2.5426     | 1.0897     |
| 6  | 8  | 2  | 1 | 0.8015  | 0.6904 | 0.3096  | 2.4767     | 0.6191     |

### 🔢 Gradient Sums

| Term        | Σ Value     |
| ----------- | ----------- |
| Σ(y - ŷ)·x₁ | **4.3198**  |
| Σ(y - ŷ)·x₂ | **-2.8118** |
| Σ(y - ŷ)    | **0.0451**  |

### 📌 **Updated Weights After Epoch 4**

$$
\begin{align*}
w₁ &= 0.1529 + 0.01 × 4.3198 = \textbf{0.1961} \\
w₂ &= -0.1306 + 0.01 × (-2.8118) = \textbf{-0.1587} \\
b  &= -0.0020 + 0.01 × 0.0451 = \textbf{-0.0016}
\end{align*}
$$

---

## 🔁 **Epoch 5 Table**

| ID | x₁ | x₂ | y | z       | ŷ      | y - ŷ   | (y - ŷ)·x₁ | (y - ŷ)·x₂ |
| -- | -- | -- | - | ------- | ------ | ------- | ---------- | ---------- |
| 1  | 2  | 7  | 0 | -1.0911 | 0.2514 | -0.2514 | -0.5028    | -1.7595    |
| 2  | 3  | 6  | 0 | -0.8948 | 0.2902 | -0.2902 | -0.8707    | -1.7413    |
| 3  | 4  | 5  | 0 | -0.6985 | 0.3321 | -0.3321 | -1.3286    | -1.6607    |
| 4  | 6  | 4  | 1 | 0.4231  | 0.6042 | 0.3958  | 2.3750     | 1.5832     |
| 5  | 7  | 3  | 1 | 0.7306  | 0.6749 | 0.3251  | 2.2757     | 0.9752     |
| 6  | 8  | 2  | 1 | 1.0381  | 0.7385 | 0.2615  | 2.0921     | 0.5230     |

### 🔢 Gradient Sums

| Term        | Σ Value     |
| ----------- | ----------- |
| Σ(y - ŷ)·x₁ | **4.0407**  |
| Σ(y - ŷ)·x₂ | **-2.0801** |
| Σ(y - ŷ)    | **0.1086**  |

### 📌 **Updated Weights After Epoch 5**

$$
\begin{align*}
w₁ &= 0.1961 + 0.01 × 4.0407 = \textbf{0.2365} \\
w₂ &= -0.1587 + 0.01 × (-2.0801) = \textbf{-0.1795} \\
b  &= -0.0016 + 0.01 × 0.1086 = \textbf{-0.0005}
\end{align*}
$$

---

## 🔁 **Epoch 6 Table**

| ID | x₁ | x₂ | y | z       | ŷ      | y - ŷ   | (y - ŷ)·x₁ | (y - ŷ)·x₂ |
| -- | -- | -- | - | ------- | ------ | ------- | ---------- | ---------- |
| 1  | 2  | 7  | 0 | -1.3543 | 0.2051 | -0.2051 | -0.4102    | -1.4355    |
| 2  | 3  | 6  | 0 | -1.1501 | 0.2404 | -0.2404 | -0.7211    | -1.4427    |
| 3  | 4  | 5  | 0 | -0.9459 | 0.2797 | -0.2797 | -1.1188    | -1.3985    |
| 4  | 6  | 4  | 1 | 0.5150  | 0.6250 | 0.3750  | 2.2500     | 1.5000     |
| 5  | 7  | 3  | 1 | 0.8993  | 0.7108 | 0.2892  | 2.0244     | 0.8677     |
| 6  | 8  | 2  | 1 | 1.2837  | 0.7829 | 0.2171  | 1.7368     | 0.4342     |

### 🔢 Gradient Sums

| Term        | Σ Value     |
| ----------- | ----------- |
| Σ(y - ŷ)·x₁ | **3.7611**  |
| Σ(y - ŷ)·x₂ | **-1.4749** |
| Σ(y - ŷ)    | **0.0561**  |

### 📌 **Updated Weights After Epoch 6**

$$
\begin{align*}
w₁ &= 0.2365 + 0.01 × 3.7611 = \textbf{0.2741} \\
w₂ &= -0.1795 + 0.01 × (-1.4749) = \textbf{-0.1942} \\
b  &= -0.0005 + 0.01 × 0.0561 = \textbf{0.0001}
\end{align*}
$$

---
