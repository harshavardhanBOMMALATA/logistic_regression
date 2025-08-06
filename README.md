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
