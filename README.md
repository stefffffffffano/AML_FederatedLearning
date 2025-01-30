# Federated Learning Project: Optimizing Client Selection with Evolutionary Algorithms  

## 📖 Overview  

Federated Learning (FL) is a decentralized machine learning approach where multiple devices train a shared model without exchanging local data. This project explores **client selection strategies** using **Evolutionary Algorithms (EAs)** to analyze their impact on model convergence and communication efficiency, comparing them with baseline methods.  

We conduct experiments on **two datasets**:  
- 📸 **CIFAR-100** – Image classification task  
- 🎭 **Shakespeare** – Character-level language modeling  

The experiments include **centralized** and **federated** settings to provide a comparative analysis of Federated Learning performance.  

---

## 📂 Repository Structure  

The repository is structured into two main dataset-specific directories:  

### 🔹 `cifar100/`  
- `data/cifar100/` → Data loading utilities  
- `models/` → Model definitions  
- `plots_centralized/`, `plots_federated/` → Experiment results  
- `trained_models/` → Saved models  
- `utils/` → Utility scripts  
- `Client.py` → Client-side logic for FL  
- `Server.py` → Server-side logic for FL  
- `federated_notebook.ipynb` → Jupyter notebook for FL experiments  
- `train_centralized.ipynb` → Jupyter notebook for centralized experiments  
- `Report.md` → Detailed results and analysis on centralized and federated baseline configurations 
- `personal_contribution/` → **Implementation of Evolutionary Algorithms (EA) for client selection** with experiment result (`plots_federated`) and Jupyter notebook (`Experiments.ipynb`)

### 🔹 `shakespeare/`  
- `LEAF_data/` → Dataset preprocessing scripts from **LEAF Benchmark**  
- `trained_models/` → Saved models  
- `plot_centralized/`, `plot_federated/` → Experiment results  
- `Client.py` → Client-side logic  
- `Server.py` → Server-side logic  
- `Model.py` → LSTM model for character prediction  
- `Centralized_Shakespeare.ipynb` → Centralized experiments  
- `Federated_Shakespeare.ipynb` → Federated experiments  
- `Report.md` → **Dataset creation information** and architecture details  
- `personal_contribution/` → **Implementation of Evolutionary Algorithms (EA) for client selection** with experiment result (`plots_federated`) and Jupyter notebook (`Experiments.ipynb`)
---

## 🎯 Personal Contribution Methodology  

We propose an **evolutionary-based client selection strategy** to tackle statistical and system heterogeneity in FL.  

 **Key Contributions:**  
✅ **Adaptive Client Selection**: Using EAs to select high-loss clients for better model convergence  
✅ **Fairness Considerations**: Ensuring diverse client participation without exposing private data  

---

## 📊 Experimental Setup  

### 🗄 **Datasets**  
1️⃣ **CIFAR-100**: Standard image classification dataset  
2️⃣ **Shakespeare** (from [LEAF Benchmark](https://leaf.cmu.edu/)): Character-level text prediction  

### 🏗 **Federated Learning Setup**  
- **Client-Server Architecture** with multiple clients training local models  
- **Heterogeneous Data Distribution** to simulate real-world FL scenarios  
- **Client Selection Strategies**:  
  - Baseline methods (Random selection, Uniform sampling)  
  - **Evolutionary Algorithm (EA)-based selection** for adaptive client participation  
- **Impact of Skewed Client Participation**:  
  - Clients selected with probabilities drawn from a **Dirichlet distribution** (γ-controlled heterogeneity)  
  - Analysis of test accuracy degradation under different γ values  

### 🔄 **Impact of Local Updates (J) and Communication Rounds**  
- **Evaluation under different numbers of local steps (J)**  
  - Trade-off analysis between local model drift and communication frequency  
  - Scaling rounds to balance global updates vs. local training  
  - Effect of J on CIFAR-100 and Shakespeare datasets  

### 🔧 **Hyperparameter Tuning**  
- **CIFAR-100**: Grid search on **learning rates, weight decays, and schedulers** (Only for Centralized version: StepLR, Cosine Annealing, Exponential LR)  
- **Shakespeare**: Preprocessing and tuning based on **LEAF Benchmark** and literature guidelines  
- **EA-based selection**: Hyperparameter tuning on **crossover probabilities**  

---

## 🚀 Getting Started  

### 🔹 **1. Clone the repository**  
```bash
git clone https://github.com/stefffffffffano/AML_FederatedLearning.git
cd AML_FederatedLearning
```

### 🔹 **2. Set up the environment**  
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install -r requirements.txt
```

### 🔹 **3. Run Experiments**  
#### 🖼 CIFAR-100  
- **Centralized Training:**  
  ```bash
  jupyter notebook cifar100/train_centralized.ipynb
  ```
- **Federated Learning:**  
  ```bash
  jupyter notebook cifar100/federated_notebook.ipynb
  ```
- **Personal Contribution (EA-based Client Selection) Notebook:**  
  ```bash
  jupyter notebook personal_contribution/Experiments.ipynb
  ```

#### 🎭 Shakespeare  
- **Centralized Training:**  
  ```bash
  jupyter notebook shakespeare/Centralized_Shakespeare.ipynb
  ```
- **Federated Learning:**  
  ```bash
  jupyter notebook shakespeare/Federated_Shakespeare.ipynb
  ```
- **Personal Contribution (EA-based Client Selection) Notebook:**  
  ```bash
  jupyter notebook personal_contribution/Experiments.ipynb
  ```

---

## 🤝 Contributions  

Contributions are welcome! Feel free to fork the repo and submit pull requests. 🚀  

---

## 📬 Contact  

**Authors:**  
- 🧑‍💻 [Luca Dadone](https://github.com/dadoluca) 
- 🧑‍💻 [Stefano Fumero ](https://github.com/stefffffffffano) 
- 🧑‍💻 [Andrea Mirenda](https://github.com/Andrea-1704) 

For any inquiries, please reach out to the authors.  

