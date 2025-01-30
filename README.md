# Optimizing Client Selection in Federated Learning with Evolutionary Algorithms  

## 📖 Overview  

Federated Learning (FL) is a decentralized approach to machine learning where models are trained across multiple devices while keeping data localized. This project investigates **client selection strategies** in FL using **Evolutionary Algorithms (EAs)** to improve convergence and reduce communication overhead.  

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
- `Report.md` → Detailed results and analysis  
- `personal_contribution/` → **Implementation of Evolutionary Algorithms (EA) for client selection (Jupyter notebook: `Experiments.ipynb`)**  

### 🔹 `shakespeare/`  
- `LEAF_data/` → Dataset preprocessing scripts from **LEAF Benchmark**  
- `trained_models/` → Saved models  
- `plot_centralized/`, `plot_federated/` → Experiment results  
- `Client.py` → Client-side logic  
- `Server.py` → Server-side logic  
- `Model.py` → LSTM model for character prediction  
- `Centralized_Shakespeare.ipynb` → Centralized experiments  
- `Federated_Shakespeare.ipynb` → Federated experiments  
- `Report.md` → Results and insights  
- `personal_contribution/` → **Implementation of Evolutionary Algorithms (EA) for client selection (Jupyter notebook: `Experiments.ipynb`)**  
---

## 🎯 Personal Contribution Methodology  

We propose an **evolutionary-based client selection strategy** to tackle statistical and system heterogeneity in FL.  

### ⚡ **Key Contributions**  
✅ **Adaptive Client Selection**: Using EAs to select high-loss clients for better model convergence  
✅ **Fairness Considerations**: Ensuring diverse client participation without exposing private data  

---

## 📊 Experimental Setup  

### 📌 **Datasets**  
1️⃣ **CIFAR-100**: Standard image classification dataset  
2️⃣ **Shakespeare** (from [LEAF Benchmark](https://leaf.cmu.edu/)): Character-level text prediction  

### 🏗 **Federated Learning Setup**  
- **Client-Server Architecture** with multiple clients updating local models  
- **Heterogeneous Data Distribution** simulating real-world FL scenarios  
- **Comparison with Baseline Strategies**:  
  - Random client selection  
  - Uniform sampling  
  - Proposed **EA-based selection**  

### 🔧 **Hyperparameter Tuning**  
- Different **learning rates (lr)**, **weight decays (wd)**, and **schedulers**  
- Details available in `Report.md`  

---

## 📈 Results  

🔹 **Federated vs. Centralized Performance**  
- Evaluated model accuracy & loss trends for different settings  
- Comparison of FL client selection strategies  

🔹 **Impact of Client Selection**  
- Improved convergence speed with EA-based selection  
- Better utilization of computational resources  

🔹 **Plots & Visualizations**  
- Accuracy & loss curves saved in `plots_federated/` and `plots_centralized/`  

---

## 🚀 Getting Started  

### 🔹 **1. Clone the repository**  
```bash
git clone https://github.com/your-repo.git
cd your-repo
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

#### 🎭 Shakespeare  
- **Centralized Training:**  
  ```bash
  jupyter notebook shakespeare/Centralized_Shakespeare.ipynb
  ```
- **Federated Learning:**  
  ```bash
  jupyter notebook shakespeare/Federated_Shakespeare.ipynb
  ```

---

## 📚 References  

1️⃣ **LEAF Benchmark**: Caldas, Sebastian, et al. *"Leaf: A benchmark for federated settings."* arXiv preprint arXiv:1812.01097 (2018).  
2️⃣ **Federated Optimization**: Reddi, Sashank, et al. *"Adaptive Federated Optimization."* 2021.  

---

## 🎯 Future Work  

🔹 Apply the method to **more complex datasets**  
🔹 Extend selection strategies with **reinforcement learning**  
🔹 Optimize for **mobile and edge computing** scenarios  

---

## 🤝 Contributions  

Contributions are welcome! Feel free to fork the repo and submit pull requests. 🚀  

---

## 🛠 Contact  

**Authors:**  
- 🧑‍💻 Luca Dadone  
- 🧑‍💻 Stefano Fumero  
- 🧑‍💻 Andrea Mirenda  

📩 For any inquiries, please reach out to the authors.  

