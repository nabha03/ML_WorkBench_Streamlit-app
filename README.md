# 🤖 ML Workbench

An interactive **Streamlit** application for running Supervised and Unsupervised Machine Learning techniques through a clean, modern UI — train, evaluate, visualize, and export results to Excel in one click.

 🔗 **Live App:** [Click here to open](https://mlworkbenchapp-app-iztxjyom6attgsqk6re7ne.streamlit.app/)

---

## ✨ Features

### 🧠 ML Techniques
| Task | Algorithms |
|------|-----------|
| Supervised — Regression | Linear Regression, Random Forest Regressor |
| Supervised — Classification | Logistic Regression, Random Forest, Decision Tree, SVM |
| Unsupervised — Clustering | K-Means, DBSCAN |
| Unsupervised — PCA | 2-Component PCA, 3-Component PCA |

### 📦 Data Sources
- **Upload your own** CSV or Excel file
- **19 Seaborn built-in datasets** — tips, titanic, iris, diamonds, penguins, mpg, flights, and more

### 📊 Visual Outputs
- Regression → Actual vs Predicted scatter + feature weights
- Classification → Confusion matrix heatmap + full classification report
- Clustering → 2D PCA scatter plot colored by cluster
- PCA → Scree plot + biplot + loadings matrix

### 📥 Excel Export (5 Sheets)
| Sheet | Contents |
|-------|----------|
| Sheet 1 | Raw Data |
| Sheet 2 | Training Set |
| Sheet 3 | Test Set |
| Sheet 4 | Model Coefficients / Feature Weights |
| Sheet 5 | Model Evaluation (metrics + confusion matrix) |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/ml-workbench.git
cd ml-workbench

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run ml_workbench.py
```

---

## 🌐 Deploy on Streamlit Community Cloud

1. Fork or push this repo to your GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select this repo
4. Set **Main file path** to `ml_workbench.py`
5. Click **Deploy** — done!

---

## 🗂️ Project Structure

```
ml-workbench/
├── ml_workbench.py      # Main Streamlit app
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **ML:** Scikit-learn
- **Data:** Pandas, NumPy, Seaborn
- **Visualization:** Matplotlib
- **Export:** OpenPyXL

---

## 👤 Author
Vaishali Kawadapure
Built as a portfolio project — part of PG Diploma in Big Data Analytics (CDAC Chennai).
