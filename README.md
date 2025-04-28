---

# 🧠 Neuroevolution-Based Language Model for Text Generation

## Overview

In this project, we developed an **automated framework** that uses **neuroevolution** to evolve optimal neural network architectures for **text generation** tasks.  
Instead of manually designing and tuning the language model, this system **evolves** both **architecture** and **hyperparameters** using **evolutionary algorithms**.

Through a systematic process involving **mutation**, **crossover**, and **fitness evaluation**, the framework discovers architectures that perform well on given text generation datasets — drastically reducing human intervention and boosting efficiency.

---

## ✨ Key Features

- **Automatic Model Discovery**:  
  Models are generated randomly and evolved over generations without manual design.

- **Flexible Architecture Search**:  
  Architectures evolve combinations of:
  - **LSTM Layers** (units vary between 700–1000)
  - **Dropout Layers** (rates between 0.1–0.3)
  - **Dense Layers** (units between 50–66)

- **Evolutionary Operations**:
  - **Mutation**: Randomly tweaks layer parameters to explore new architectures.
  - **Crossover**: Swaps parts of two models to combine strengths.
  - **Selection**: Chooses best-performing architectures based on evaluation fitness.

- **Fitness Evaluation**:  
  Models are trained and evaluated based on their validation **accuracy**; lower error corresponds to higher fitness.

- **No Manual Hyperparameter Tuning**:  
  The system searches and optimizes learning rates, layer types, and structure automatically.

- **Implemented Entirely in Jupyter Notebook**:  
  The project is structured as an `.ipynb` notebook for easy step-by-step execution and experimentation.

---

## 🏗️ Project Architecture

| Component        | Description                                                                          |
|------------------|--------------------------------------------------------------------------------------|
| **Generating Individuals** | Randomly generates LSTM-based model blueprints with random units/dropout |
| **Model Builder** | Converts individuals (blueprints) into actual Keras models using TensorFlow         |
| **Crossover**     | Swaps parts of two models (LSTM or Dropout blocks) to create offspring              |
| **Mutation**      | Randomly modifies parts of a model to introduce variability                        |
| **Evaluation**    | Trains and validates each model, returning fitness (1 / validation accuracy)        |
| **Evolution Loop**| Repeats selection, crossover, mutation, and evaluation for multiple generations     |

---

## 📜 Research Publication

I'm thrilled to announce that the work based on this project has been published!

> **Title:** Neuro-Evolution-Based Language Model for Text Generation  
> **Conference:** International Conference on Computational Intelligence in Data Science (**ICCIDS 2024**)  
> **Book Series:** IFIP Advances in Information and Communication Technology (**AICT**)  
>  
> This research introduces a novel method where **Genetic Algorithms** evolve **LSTM architectures** — **eliminating the need for manual architecture tuning** in NLP tasks.  
>  
> 🔗 **Paper Link:** [Read the full paper here](https://lnkd.in/g727Rp7s)

A heartfelt thank you to **Bagavathi Shivakumar** for her invaluable guidance, mentorship, and encouragement throughout this journey.

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow / Keras**
- **DEAP** (Distributed Evolutionary Algorithms in Python)
- **NumPy**
- **Matplotlib** (for visualizations)
- **Jupyter Notebook**

---

## ⚙️ Important Notes

- The project runs inside a **Jupyter Notebook** (`.ipynb` format).
- It uses TensorFlow datasets (`train_dataset`, `test_dataset`) for training and validation.
- The evolutionary parameters such as `MU` (population size), `NGEN` (number of generations), and `MUTPB` (mutation probability) can be customized easily.
- All training, evaluation, and evolutionary progress are logged inside the notebook for monitoring.

---

# 🚀 Future Work
- Extend neuroevolution to **Transformer**-based architectures.
- Introduce **multi-objective optimization** (e.g., accuracy vs. model size).
- Experiment with **different fitness functions** (e.g., BLEU score for text quality).

---

---

# ✅ Quick Project Summary (as you also asked)

> This project builds a **Neuroevolutionary system** for automatically generating the best-performing **LSTM-based language models** for text generation.  
Instead of manual model design, it uses **evolutionary algorithms** to **mutate, crossover, and evolve** models based on validation performance.  
This research was recognized and **published** at **ICCIDS 2024** under the **IFIP AICT** series.

---
