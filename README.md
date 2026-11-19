# 🎓 Student Final Score Analysis & Predictive Modeling

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Poppins&weight=700&size=30&duration=3500&pause=1000&color=00BFFF&center=true&vCenter=true&width=950&lines=Student+Final+Score+Analysis+%26+Prediction;Data+Cleaning+%7C+EDA+%7C+Regression+Modeling;Leakage-Free+End-to-End+ML+Pipeline" alt="Typing SVG" />
</p>

<p align="center">
  <b>📊 Exploratory Data Analysis • 🤖 Supervised Machine Learning • 📈 Statistical Modeling • 🔍 Leakage-Free Pipeline</b>
</p>

<p align="center">
  <i>"Consistency in attendance, study habits, and continuous assessments shapes academic outcomes—data reveals how."</i>
</p>

---

## 📖 Project Overview

Academic achievement is driven by a combination of classroom attendance, weekly self-study habits, continuous assessments (assignments and quizzes), sleep balance, and prior academic foundation. However, real-world student records often contain duplicate entries, out-of-range measurements, missing values, and post-outcome target leakage.

This project builds an **end-to-end, leakage-free Exploratory Data Analysis (EDA) and Regression Modeling pipeline** to investigate the key drivers of student performance and accurately predict `final_score`.

The analysis explores patterns involving:

* 📅 Attendance Percentage (`attendance_pct`)
* 📚 Weekly Study Hours (`study_hours_week`)
* 📝 Assignment Average (`assignment_avg`)
* ✍️ Quiz Average (`quiz_avg`)
* 😴 Sleep Duration (`sleep_hours`)
* 🎯 Prior Academic Score (`prior_score`)
* 💻 Study Mode (`study_mode`)
* 🧭 Course Track (`course_track`)

Rather than treating modeling as a black-box exercise, this project emphasizes **domain-driven data sanitization, strict train/validation/test isolation, model benchmarking, and residual diagnostics**.

---

# 🎯 Research & Modeling Objective

> **Which academic, behavioral, and course-level factors most strongly influence a student's `final_score`, and which regression algorithm generalizes best to unseen student records without data leakage?**

The objective is to build a reproducible predictive workflow while maintaining strict separation between training statistics and evaluation splits.

---

# 🌍 Why This Project Matters

Educational institutions and educators benefit from early, reliable indicators of student performance. Identifying how continuous coursework, attendance, and study habits relate to final exam scores enables timely academic intervention and data-backed study planning.

This project demonstrates how **structured data engineering and disciplined ML evaluation** turn noisy student records into dependable predictive insights:

* 🧹 Domain-aware anomaly filtering
* 🔒 Strict zero-leakage preprocessing
* 📊 Statistical & visual relationship analysis
* ⚖️ Fair multi-model benchmarking
* 📉 Residual error & bias verification

---

# 📂 Dataset Overview

The dataset (`DATASET.csv`) contains individual student academic records:

* **Initial Raw Shape:** `708 rows × 12 columns`
* **Cleaned Modeling Shape:** `700 rows × 9 columns` (after deduplication and target filtering)
* **Target Variable:** `final_score` (Continuous: `46.49` to `100.00`, Mean: `80.38`)

### Features Included

| Feature | Type | Analytical Role |
| :--- | :---: | :--- |
| `attendance_pct` | Numerical | Measures classroom participation (`0–100%`) |
| `study_hours_week` | Numerical | Captures weekly self-study effort (`0–168 hrs`) |
| `assignment_avg` | Numerical | Reflects continuous coursework consistency |
| `quiz_avg` | Numerical | Measures periodic conceptual retention |
| `sleep_hours` | Numerical | Captures daily recovery and rest balance |
| `prior_score` | Numerical | Represents baseline academic readiness |
| `study_mode` | Categorical | Compares learning delivery modes |
| `course_track` | Categorical | Captures academic track/specialization differences |

### 🚫 Excluded Columns (Leakage & Noise Prevention)

* **`record_id`**: Identifier column with zero predictive value.
* **`final_grade_letter`**: Directly derived from `final_score`—removed to prevent **target leakage**.
* **`random_noise_feature`**: Synthetic noise feature dropped to avoid spurious splits and overfitting.

---

# 🔍 Exploratory Data Analysis & Preprocessing

## 🧹 1. Data Understanding & Sanity Filtering

Reliable modeling begins with clean, domain-valid records.

### Steps Performed

* Inspected dataset shape, column data types, summary statistics, and missing value counts
* Removed **8 exact duplicate rows** and filtered missing target entries (`708 → 700 rows`)
* Coerced impossible/out-of-range values to `NaN` (`attendance_pct` outside `0–100`, `study_hours_week` outside `0–168`, negative scores/hours)
* Standardized text casing and whitespace for categorical features (`study_mode`, `course_track`)

---

## 🔒 2. Leakage-Free Data Splitting & Transformation

To ensure realistic evaluation metrics, preprocessing statistics were isolated strictly within the training split:

```text
Cleaned Dataset (700 rows)
   ├──► Training Set   : 489 rows (~70%) ──► Fit Medians, Modes & Encoding Schema
   ├──► Validation Set : 106 rows (~15%) ──► Model Selection & Benchmarking
   └──► Test Set       : 105 rows (~15%) ──► Final Unseen Evaluation
```

* Missing numerical values imputed using **training medians**
* Missing categorical values imputed using **training modes**
* One-hot encoding (`pd.get_dummies`) aligned strictly across train, validation, and test splits

---

## 📈 3. Univariate & Bivariate Exploration

### Visualizations & Diagnostics

* 📊 Target distribution histogram & summary statistics (`final_score` mean `80.38`, range `46.49 – 100.00`)
* 🔵 Bivariate scatter plots comparing each numerical predictor against `final_score`
* 🔥 Correlation heatmap across numerical predictors to assess linear strength and multicollinearity

---

# 🤖 Model Benchmarking & Evaluation

Four regression algorithms were trained on the `489` training records and evaluated on the `106` validation records using **Validation RMSE** as the primary selection criterion:

| Rank | Model | Validation MAE | Validation RMSE | Validation R² | Generalization Behavior |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 🥇 **1** | **Linear Regression** | **3.126** | **4.492** | **0.834** | **Best overall fit & lowest validation error** |
| 🥈 **2** | **Gradient Boosting** | 3.553 | 4.537 | 0.831 | Strong runner-up; captures subtle non-linearities |
| 🥉 **3** | **Random Forest** | 4.269 | 5.402 | 0.760 | Moderate variance on medium-sized tabular data |
| **4** | **Decision Tree** | 6.636 | 8.281 | 0.437 | High variance / prone to overfitting |

---

# 🏆 Final Test Performance & Residual Diagnostics

After winning validation selection, **Linear Regression** was retrained on the combined **Train + Validation (`595 rows`)** dataset and evaluated once on the held-out **Test Set (`105 rows`)**:

| Metric | Test Set Score | Interpretation |
| :--- | :---: | :--- |
| **Test MAE** | **`2.881`** | Predictions are within **~2.9 marks** on average |
| **Test RMSE** | **`4.500`** | Matches validation RMSE (`4.492`), proving strong stability |
| **Test R²** | **`0.835`** | Explains **83.5%** of variance in unseen student final scores |
| **Mean Residual** | **`-0.320`** | Near-zero bias (`Actual - Predicted`), well-calibrated predictions |

* **Actual vs. Predicted Plot:** Confirms tight alignment along the ideal diagonal across score ranges.
* **Residual Plot:** Shows symmetric, homoscedastic scatter around zero with no systematic drift.

---

# 💡 Key Analytical Questions & Takeaways

* 📚 **Why did Linear Regression outperform tree ensembles?** Because final academic scores are fundamentally weighted combinations of continuous performance components (`prior_score`, `assignment_avg`, `quiz_avg`, `attendance_pct`), a linear model captures the underlying additive structure cleanly without overfitting `700` rows.
* 🔒 **Why does split-before-impute matter?** Computing medians, modes, and dummy columns solely on the training split prevented data leakage, reflected in nearly identical Validation (`0.834`) and Test (`0.835`) R² scores.
* 🚫 **Why drop derived columns?** Removing `final_grade_letter` ensured the model learned genuine behavioral predictors rather than trivially reversing a grade mapping.

---

# 🚀 Project Workflow

```text
        📂 Raw Dataset (708 × 12)
                 │
                 ▼
     🧹 Deduplication & Sanity Rules
                 │
                 ▼
       📊 EDA & Correlation Analysis
                 │
                 ▼
   🔒 Train / Val / Test Split (489/106/105)
                 │
                 ▼
   ⚙️ Leakage-Free Imputation & Encoding
                 │
                 ▼
     🤖 Multi-Model Training & Validation
                 │
                 ▼
   🏆 Retrain Best Model (Train + Val)
                 │
                 ▼
    📉 Final Test & Residual Diagnostics
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
| :--- | :--- |
| 🐍 **Python** | Core Programming & Modeling |
| 🐼 **Pandas** | Data Cleaning & Tabular Manipulation |
| 🔢 **NumPy** | Numerical Operations & Array Handling |
| 📊 **Matplotlib** | Data & Diagnostic Plotting |
| 🎨 **Seaborn** | Statistical Visualizations & Heatmaps |
| 🤖 **Scikit-Learn** | Regression Models & Evaluation Metrics |
| 📓 **Jupyter Notebook** | Interactive Analysis Environment |

---

# 🎯 Skills Demonstrated

This project demonstrates practical skills in:

* 🔍 Exploratory Data Analysis (EDA)
* 🧹 Domain-Driven Data Cleaning & Validation
* 🔒 Leakage-Free Preprocessing Pipelines
* 📊 Statistical & Correlation Analysis
* 🤖 Supervised Regression Modeling
* ⚖️ Model Benchmarking & Selection
* 📉 Residual & Diagnostic Error Analysis
* 🐍 Python & Scikit-Learn Workflows
* 📈 Data Storytelling & Documentation

---

# 🌟 Project Highlights

```text
✅ Leakage-Free Train / Val / Test Split
✅ Domain-Rule Outlier & Range Filtering
✅ Target Leakage & Noise Column Removal
✅ Correlation & Bivariate EDA Diagnostics
✅ 4-Model Regression Benchmarking
✅ Retraining on Combined Train + Val Data
✅ Held-Out Test & Residual Verification
✅ Clean & Reproducible Documentation
```

---

# 🚀 Future Scope

This project can be extended through:

* 🎛️ Regularized regression tuning (`Ridge`, `Lasso`, `ElasticNet`) with `K-Fold` cross-validation
* 🧠 SHAP & feature importance analysis to quantify exact mark contributions per feature
* 🌐 Interactive **Streamlit** web application for live student score prediction
* 📦 Exporting a unified `sklearn.pipeline.Pipeline` artifact for production inference

---

# 📁 Repository Structure

```text
STUDENT_FINAL_SCORE_ANALYSIS/
│
├── 📄 DATASET.csv
│
├── 📓 Neeraj_Danu_Solutionn.ipynb
│
├── 📄 LICENSE
│
└── 📄 README.md
```

---

# 📦 Requirements & Quickstart

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

### Clone & Run Locally

```bash
# 1. Clone the repository
git clone [https://github.com/NEERAJSINGHDANU07/STUDENT_FINAL_SCORE_ANALYSIS.git](https://github.com/NEERAJSINGHDANU07/STUDENT_FINAL_SCORE_ANALYSIS.git)
cd STUDENT_FINAL_SCORE_ANALYSIS

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch Jupyter Notebook
jupyter notebook Neeraj_Danu_Solutionn.ipynb
```

---

# 👨‍💻 Author

## **Neeraj Singh Danu**

<p align="center">
  <b>Data Science • Machine Learning • Python • SQL • EDA • Data Visualization • MLOps</b>
</p>

Passionate about **Data Science, Machine Learning, and Analytics**, with a focus on transforming raw data into meaningful insights and building practical data-driven solutions.

---

<p align="center">
  ⭐ <b>If you found this project useful, consider giving the repository a Star!</b> ⭐
</p>

<p align="center">
  <b>Built with 🐍 Python & 📊 Data Science</b>
</p>
