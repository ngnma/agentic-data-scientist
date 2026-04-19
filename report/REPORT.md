# Agentic Data Scientist: An Autonomous Machine Learning Pipeline

## 1. Introduction

The rapid growth of machine learning applications has created a strong demand for systems that can automate the end-to-end data science workflow. Traditional machine learning pipelines require significant human intervention for tasks such as data preprocessing, feature engineering, model selection, evaluation, and iterative improvement. This project presents an **Agentic Data Scientist**, a system designed to autonomously perform these tasks through a structured, modular, and adaptive pipeline.

The primary goal of this project is to design and implement an intelligent system capable of making data-driven decisions at each stage of the machine learning lifecycle. Unlike static pipelines, this system incorporates a feedback loop that enables it to **reflect on its performance and iteratively improve its decisions**, mimicking the behavior of a human data scientist.

The system integrates multiple components including data profiling, planning, preprocessing, modelling, evaluation, and reflection. It also introduces a **memory mechanism** that allows the system to learn from past experiences and improve future decision-making. This report provides a comprehensive overview of the system architecture, methodology, design decisions, and evaluation of its effectiveness.

---

## 2. System Overview

The Agentic Data Scientist is structured as a **modular pipeline composed of interconnected stages**, each responsible for a specific part of the machine learning workflow. The system operates in iterative cycles, where each iteration consists of executing a plan, evaluating results, and deciding whether to replan.

The main components of the system are:

- Data Profiling
- Planning Agent
- Preprocessing Pipeline
- Model Selection and Training
- Evaluation Module
- Reflector (Feedback Mechanism)
- Memory System

The pipeline begins with loading a dataset and automatically inferring the target variable. The data is then profiled to extract statistical and structural information. Based on this profile, the planner constructs an initial plan consisting of preprocessing and modelling steps.

After executing the plan, the system evaluates the results and uses the reflector to identify potential issues and suggest improvements. If necessary, the system replans and executes another iteration. This process continues until no further improvements are deemed beneficial or a predefined limit is reached.

---

## 3. Architecture

The architecture follows an **agent-based design**, where each component acts as a specialized agent responsible for a specific task. The system maintains a shared `state` object that carries information between steps, ensuring consistency and modularity.

### 3.1 Data Flow

The overall data flow can be summarized as:

1. Load dataset
2. Profile dataset
3. Generate plan
4. Execute preprocessing and modelling
5. Evaluate performance
6. Reflect and update strategy
7. Repeat if necessary

Each step updates the shared state, allowing subsequent steps to make informed decisions.

### 3.2 Modularity

The system is designed to be highly modular:

- Each step is implemented as a separate function
- Steps can be dynamically added or removed from the plan
- Components communicate through structured dictionaries

This modularity enables flexibility, extensibility, and easier debugging.

---

## 4. Data Profiling

The data profiling module is responsible for extracting key characteristics of the dataset that influence downstream decisions.

### 4.1 Extracted Features

The profiling step computes:

- Dataset shape (rows and columns)
- Feature types (numeric, categorical, boolean)
- Missing value percentages
- Skewness of numeric features
- Outlier ratios
- Number of unique values per column
- Class imbalance (for classification tasks)

### 4.2 Design Decisions

A critical design decision was to **separate boolean features from numeric features**, as treating boolean values as continuous variables leads to errors in statistical computations. This distinction ensures that appropriate preprocessing techniques are applied.

Another important aspect is the use of **lightweight statistical summaries**, which provide sufficient information for planning without introducing unnecessary computational overhead.

---

## 5. Planning Strategy

The planner is responsible for generating a sequence of actions based on the dataset profile.

### 5.1 Rule-Based Planning

The planner uses a **rule-based approach**, where specific conditions trigger the inclusion of certain steps. For example:

- High missing values → add imputation steps
- High skewness → apply transformations
- Class imbalance → apply class weighting or resampling
- High cardinality categorical features → apply target encoding

### 5.2 Dynamic Plan Construction

The plan is constructed dynamically and sorted to ensure consistency. Duplicate steps are removed to maintain efficiency.

### 5.3 Advantages

- Transparent decision-making
- Easy to extend with new rules
- Deterministic and reproducible behavior

---

## 6. Preprocessing Pipeline

The preprocessing module transforms raw data into a suitable format for modelling.

### 6.1 Numeric Features

Numeric features undergo:

- Missing value imputation (median)
- Scaling (standardization)
- Optional transformations:
  - Yeo-Johnson (for skewness)
  - Square root transformation
  - Outlier clipping

### 6.2 Categorical Features

Categorical features are processed using:

- One-hot encoding
- Target encoding (for high cardinality)
- Ordinal encoding (when applicable)
- Frequency encoding

### 6.3 Boolean Features

Boolean features are handled separately by converting them into string representations before encoding. This avoids compatibility issues with preprocessing tools.

### 6.4 Design Considerations

The preprocessing pipeline is implemented using a **ColumnTransformer**, allowing different transformations to be applied to different feature groups simultaneously.

---

## 7. Model Selection and Training

### 7.1 Candidate Models

The system supports multiple models, including:

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Machines
- Dummy baseline models

### 7.2 Model Selection Strategy

The planner selects models based on dataset characteristics such as size and feature dimensionality.

### 7.3 Training Process

Models are trained using:

- Train-test split
- Cross-validation
- Grid search (for hyperparameter tuning)

Even when no hyperparameters are provided, cross-validation ensures robust evaluation.

### 7.4 Design Decisions

- Use of pipelines ensures preprocessing is included in cross-validation
- Inclusion of baseline models enables performance comparison

---

## 8. Evaluation

The evaluation module assesses model performance using multiple metrics:

- Accuracy
- Balanced accuracy
- F1-score (macro)
- Precision and recall
- Confusion matrix

### 8.1 Importance of Multiple Metrics

Using multiple metrics ensures that performance is evaluated comprehensively, particularly in imbalanced datasets.

### 8.2 Reporting

The system generates structured outputs including:

- JSON summaries
- Confusion matrix visualization
- Markdown reports

---

## 9. Reflection and Replanning

The reflector is the core component that enables adaptive behavior.

### 9.1 Functionality

The reflector:

- Identifies issues (e.g., overfitting, underfitting, data quality problems)
- Generates suggestions for improvement
- Determines whether replanning is necessary

### 9.2 Decision Logic

The decision to replan is based on:

- Model performance relative to target thresholds
- Confidence in suggestions
- Resource budget
- Diminishing returns
- Past outcomes stored in memory

### 9.3 Adaptive Behavior

This mechanism allows the system to:

- Avoid unnecessary iterations
- Focus on meaningful improvements
- Prevent repeated failures

---

## 10. Memory System

### 10.1 Purpose

The memory system enables the agent to learn from past runs.

### 10.2 Stored Information

- Detected issues
- Applied actions
- Performance before and after changes
- Improvement metrics

### 10.3 Usage

Memory is used to:

- Prioritize effective strategies
- Avoid repeating failed actions
- Improve confidence estimation

### 10.4 Design Benefits

- Enables meta-learning
- Improves long-term performance
- Adds persistence across runs

---

## 11. Workflow

The system operates in iterative cycles:

1. Initial plan generation
2. Execution of preprocessing and modelling
3. Evaluation of results
4. Reflection and analysis
5. Decision to replan or stop

Each iteration refines the pipeline, leading to progressively improved performance.

---

## 12. Challenges and Solutions

### 12.1 Data Type Issues

Problem: Boolean and mixed-type columns caused preprocessing errors  
Solution: Separate handling and type conversion

### 12.2 Overfitting and Underfitting

Problem: Models performed poorly due to imbalance or complexity  
Solution: Reflection-based detection and targeted adjustments

### 12.3 Computational Cost

Problem: Large datasets slowed down training  
Solution: Efficient pipeline design and selective transformations

### 12.4 Unrealistic Performance

Problem: Suspiciously high accuracy due to potential leakage  
Solution: Improved encoding strategies and feature filtering

---

## 13. Evaluation of the System

The system was tested on multiple datasets across different domains, including:

- Music classification
- Educational data
- Business datasets
- Scientific datasets

### 13.1 Observations

- The system performs well on structured tabular data
- Reflection improves results in complex scenarios
- Memory enhances decision-making over time

### 13.2 Limitations

- Rule-based planner lacks flexibility in unseen scenarios
- No deep learning models included
- Limited support for time-series data

---

## 14. Future Work

Potential improvements include:

- Integration of learning-based planning
- Support for deep learning models
- Advanced feature engineering techniques
- Better handling of large-scale datasets
- Improved interpretability of decisions

---

## 15. Conclusion

This project demonstrates the feasibility of building an autonomous data science system capable of handling the full machine learning pipeline. By combining structured planning, adaptive reflection, and memory-based learning, the Agentic Data Scientist achieves a balance between automation and intelligent decision-making.

The modular design allows for easy extension, while the reflection mechanism ensures continuous improvement. Although there are limitations, the system provides a strong foundation for future research in automated machine learning and intelligent agents.

Overall, this work highlights the potential of agentic systems to transform how data science workflows are executed, reducing manual effort while maintaining high-quality results.