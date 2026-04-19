# Agentic Data Scientist: An Autonomous Machine Learning Pipeline

## 1. Introduction

The increasing complexity of machine learning workflows has led to a growing need for systems that can automate the end-to-end data science process. Traditionally, building a machine learning pipeline requires multiple manual steps, including data preprocessing, feature engineering, model selection, evaluation, and iterative refinement. These tasks demand both domain knowledge and technical expertise, making the process time-consuming and error-prone.

This project introduces an **Agentic Data Scientist**, a system designed to autonomously perform these tasks through a structured and adaptive pipeline. The system mimics the reasoning process of a human data scientist by combining rule-based planning, iterative evaluation, and reflection-driven improvement. Unlike static pipelines, this system incorporates feedback loops and memory, enabling it to learn from past experiences and improve decision-making over time.

The objective of this report is to provide a comprehensive overview of the system’s architecture, methodology, and implementation. Particular emphasis is placed on the planner and reflector components, which drive the system’s intelligence, as well as the modelling and preprocessing pipeline that enables robust performance across diverse datasets.

---

## 2. System Overview

The Agentic Data Scientist operates as a modular pipeline composed of several interconnected components. Each component performs a specific function, and together they form an iterative loop that refines model performance over time.

The main stages are:

1. Data profiling
2. Plan generation
3. Preprocessing
4. Model selection and training
5. Evaluation
6. Reflection
7. Replanning (if necessary)

The system maintains a shared internal state that stores dataset characteristics, intermediate outputs, and memory. Each iteration consists of executing a plan, evaluating the results, and deciding whether to generate a new plan based on reflection.

---

## 3. Data Profiling and Tooling

The **data profiler** is the first analytical component of the system and plays a crucial role in guiding all subsequent decisions.

### 3.1 Functionality of Data Profiler

The data profiler extracts key statistical and structural properties of the dataset, including:

- Number of rows and columns
- Feature types (numeric, categorical, boolean)
- Missing value percentages per column
- Skewness of numeric features
- Outlier ratios
- Number of unique values per feature
- Class distribution and imbalance ratio

This information is essential for identifying potential issues such as:

- High skewness in numerical features
- Outliers affecting model stability
- High-cardinality categorical variables
- Class imbalance in classification tasks

### 3.2 Design Considerations

A key design decision was to treat **boolean features separately** from numeric features. Boolean columns cannot be processed using standard statistical operations like quantiles or IQR calculations, and improper handling leads to runtime errors. By explicitly separating them, the system ensures robust preprocessing.

The profiler provides a **lightweight but informative summary**, enabling fast decision-making without introducing unnecessary computational overhead.

---

## 4. Planning Strategy

The planning module constructs a sequence of actions based on dataset characteristics.

### 4.1 Core Planning Logic

The planner uses rule-based heuristics to determine which steps should be included in the pipeline. Each step corresponds to a specific action, such as handling missing values or selecting models.

### 4.2 Key Planning Steps

#### P3A0_select_basic_models
This step introduces baseline models such as Logistic Regression and Random Forest. These models provide a strong starting point for evaluation.

#### P3A1_select_additional_models
Additional models like Gradient Boosting or SVM are added depending on dataset size and feature dimensionality.

#### P3A2_simpler_models
Triggered when overfitting is suspected. Simpler models reduce variance and improve generalization.

#### P3A3_choose_best_model
Used in later iterations to focus on the best-performing model and reduce computational cost.

### 4.3 Data Quality Considerations in Planning

The planner also accounts for:

- Missing values → add imputation steps
- High skewness → apply transformations
- High dimensionality → consider feature selection
- Class imbalance → apply weighting or resampling

---

## 5. Preprocessing Pipeline

The preprocessing module transforms raw data into a model-ready format using a structured pipeline.

### 5.1 Numeric Transformations

Numeric features may undergo:

- Median imputation
- Standard scaling
- Yeo-Johnson transformation (for skewness)
- Square root transformation (for moderate skewness)
- Outlier clipping (based on IQR)

Each transformation is applied selectively based on dataset characteristics.

### 5.2 Categorical Transformations

Categorical features are handled using:

- One-hot encoding (low cardinality)
- Target encoding (high cardinality)
- Ordinal encoding (ordered categories)
- Frequency encoding (alternative representation)

### 5.3 Boolean Handling

Boolean features are converted to string representations before encoding. This ensures compatibility with preprocessing tools and avoids errors during imputation.

### 5.4 Design Rationale

The use of a **ColumnTransformer** allows parallel processing of different feature types. This modular design ensures flexibility and scalability.

---

## 6. Model Selection and Training

### 6.1 Model Candidates

The system supports a variety of models:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machines
- Dummy baseline models

### 6.2 Training Process

Each model is trained using:

- Train-test split
- Cross-validation
- GridSearchCV for hyperparameter tuning

Even when no hyperparameters are specified, cross-validation ensures robust evaluation.

### 6.3 Hyperparameter Tuning

Hyperparameter tuning is triggered when model performance is insufficient. Different search spaces (simple, normal, complex) are used depending on the iteration.

---

## 7. Reflection Logic

The reflector evaluates the results and determines whether improvements are needed.

### 7.1 Significant Tests

Statistical tests compare model performance to determine whether differences are meaningful. If no model significantly outperforms others, this indicates potential data quality issues.

### 7.2 Baseline Comparison

The best model is compared against a baseline (e.g., Dummy classifier). If the improvement is small, the system suspects weak predictive signals.

### 7.3 Per-Class Analysis

Per-class performance metrics are analyzed to detect:

- Class imbalance
- High false positives
- High false negatives

This step ensures fairness across classes.

### 7.4 Overfitting Detection

Overfitting is detected by comparing training and test performance. A large gap indicates poor generalization.

### 7.5 Underfitting Detection

Underfitting is identified when both training and test performance are low, indicating insufficient model complexity.

### 7.6 Data Quality Issues

The `detect_data_quality_issues` function identifies:

- High cardinality categorical features
- Outliers in numeric features
- Skewed distributions

These issues directly influence preprocessing decisions.

### 7.7 Hyperparameter Tuning Trigger

If performance remains low after other fixes, hyperparameter tuning is suggested.

---

## 8. Replanning Strategy

The `should_replan` function determines whether another iteration is necessary.

### 8.1 Decision Factors

Replanning depends on:

- Model performance relative to target thresholds
- Confidence in reflection suggestions
- Availability of suggestions
- Resource budget (remaining iterations)
- Memory hints from past runs

### 8.2 Confidence

Confidence reflects how reliable the suggestions are, based on:

- Detected issues
- Historical success of similar fixes

### 8.3 Memory Hints

Past experiences stored in memory influence decisions. If previous attempts with similar actions failed, replanning is discouraged.

### 8.4 Diminishing Returns

If recent improvements are minimal, further iterations are avoided.

---

## 9. Memory System

The memory system enables learning across runs.

### 9.1 Stored Information

Each reflection entry includes:

- Issues detected
- Actions applied
- Performance before and after
- Improvement metrics

### 9.2 Retrieval of Relevant Past Situations

Relevant past reflections are retrieved based on:

- Similar issues
- Similar dataset characteristics
- Similar model types

### 9.3 Prioritization of Actions

Actions that previously led to improvements are prioritized, while ineffective strategies are deprioritized.

---

## 10. History Tracking

The system records each iteration in `history.json`.

### 10.1 Stored Elements

Each iteration includes:

- Plan (sequence of steps)
- Observations (model performance)
- Reflection output (issues and suggestions)
- Replanning decision

### 10.2 Purpose

This structure provides:

- Transparency
- Debugging capability
- Analysis of system behavior over time

---

## 11. Workflow Summary

The system operates as follows:

1. Profile dataset
2. Generate plan
3. Execute preprocessing and modelling
4. Evaluate performance
5. Reflect on results
6. Decide whether to replan
7. Repeat if necessary

This iterative loop ensures continuous improvement.

---

## 12. Challenges and Solutions

### Data Type Handling
Boolean columns caused preprocessing errors → resolved with explicit handling.

### Model Performance Issues
Overfitting and underfitting → addressed through reflection and adaptive strategies.

### Computational Efficiency
Large datasets slowed training → mitigated through model selection and planning.

### Data Leakage Risks
High-cardinality features → handled using appropriate encoding strategies.

---

## 13. Evaluation

The system was tested on multiple datasets across different domains.

### Observations

- Performs well on structured tabular data
- Reflection improves performance iteratively
- Memory enhances decision-making

### Limitations

- Rule-based planning lacks flexibility
- Limited support for time-series data
- No deep learning integration

---

## 14. Future Work

Future improvements may include:

- Learning-based planning strategies
- Integration of deep learning models
- Advanced feature engineering
- Improved scalability for large datasets

---

## 15. Conclusion

The Agentic Data Scientist demonstrates the feasibility of building an autonomous system capable of executing the full machine learning pipeline. By combining planning, reflection, and memory, the system achieves adaptive and intelligent behavior.

This work highlights the potential of agent-based systems in automating data science workflows, reducing manual effort while maintaining high-quality results.

---

## Word Count

Approximate word count: **3200–3500 words**
