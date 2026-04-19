# 🤖 Agentic Data Scientist

An autonomous, modular machine learning pipeline that mimics the workflow of a human data scientist.
The system profiles data, builds a plan, trains models, evaluates performance, and iteratively improves results through reflection and memory.

---

## 🚀 Overview

The **Agentic Data Scientist** is designed to automate the end-to-end machine learning process on tabular datasets.
It combines rule-based planning with adaptive feedback (reflection) and memory to continuously improve model performance across iterations.

Unlike static pipelines, this system:

* Analyzes dataset characteristics
* Dynamically builds preprocessing and modelling steps
* Evaluates results using multiple metrics
* Reflects on performance and suggests improvements
* Learns from past runs using memory

---

## ✨ Features

* 🔍 **Automatic Data Profiling**

  * Detects feature types, missing values, skewness, outliers, and class imbalance

* 🧠 **Intelligent Planning**

  * Generates a dynamic pipeline based on dataset characteristics

* ⚙️ **Flexible Preprocessing**

  * Numeric transformations (scaling, Yeo-Johnson, clipping)
  * Multiple encoding strategies (one-hot, target, ordinal, frequency)
  * Robust handling of boolean features

* 🤖 **Model Selection & Training**

  * Supports multiple models (Logistic Regression, Random Forest, SVM, etc.)
  * Uses cross-validation and optional hyperparameter tuning

* 📊 **Evaluation & Reporting**

  * Generates metrics, confusion matrix, and markdown reports

* 🔁 **Reflection & Replanning**

  * Detects issues such as overfitting, underfitting, and data quality problems
  * Suggests improvements and triggers replanning

* 🧠 **Memory-Based Learning**

  * Stores past reflections and outcomes
  * Prioritizes effective strategies in future runs

---

## 🏗️ Project Structure

```bash
.
├── agents/                # Core logic (planner, reflector, memory)
├── tools/                 # Data profiling, preprocessing, modelling, evaluation
├── data/                  # Sample datasets
├── report/                # Project report and documentation
├── tests/                 # Unit tests (pytest)
├── outputs/               # Generated results (reports, metrics, plots)
├── run_agent.py           # Entry point for running the system
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:

```bash
git clone https://github.com/ngnma/agentic-data-scientist.git
cd agentic-data-scientist
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the pipeline on a dataset:

```bash
python run_agent.py --data data/example_dataset.csv --target auto
```

### Options

* `--data` : Path to dataset
* `--target` : Target column (`auto` for automatic detection)
* `--output_root` : Output directory (optional)
* `--quiet` : Reduce logs

---

## 🔁 Workflow

The system follows an iterative pipeline:

1. **Profile Dataset**
2. **Generate Plan**
3. **Preprocess Data**
4. **Train Models**
5. **Evaluate Performance**
6. **Reflect on Results**
7. **Replan if Needed**

Each iteration improves the pipeline until performance stabilizes.

---

## 🧠 Memory System

The system stores past runs in memory to:

* Avoid repeating failed strategies
* Prioritize successful actions
* Improve decision-making over time

---

## 📊 Outputs

After each run, the system generates:

* `history.json` → Iteration logs (plans, metrics, reflections)
* `report.md` → Human-readable report
* `confusion_matrix.png` → Visualization
* Model evaluation metrics

---

## 🧪 Testing

Run tests with coverage:

```bash
pytest --cov=agents --cov=tools --cov-report=html tests/
```

---

## 📚 Datasets

Sample datasets are provided in the `data/` folder.
See `data/README.md` for details.

---

## ⚠️ Limitations

* Designed primarily for tabular data
* Rule-based planning (not fully learned)
* Limited support for time-series and deep learning models
* Only covers classification task

---

## 🔮 Future Improvements

* Learning-based planning strategies
* Integration with deep learning frameworks
* Advanced feature engineering
* Improved scalability for large datasets

---

## 👤 Author

* Negin (MSc Artificial Intelligence)

---

## 📄 License

This project is for academic purposes. Add a license if distributing publicly.
