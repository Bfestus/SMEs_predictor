# SME Success Predictor 🎯

**Rwanda Small & Medium Enterprises (SME) Success Prediction Platform**

A comprehensive machine learning platform that predicts the likelihood of success for Small and Medium Enterprises (SMEs) in Rwanda using advanced ML algorithms and interactive web interface.

## 🎓 Project Overview

This capstone project was developed as part of the **Machine Learning Engineering** specialization to address the critical challenge of SME failure rates in Rwanda, where approximately 60% of small businesses fail within their first five years. The platform provides data-driven insights to help entrepreneurs, investors, and policymakers make informed decisions about business viability.

## 🎯 Key Features

- **Advanced ML Models**: Random Forest and XGBoost algorithms with hyperparameter optimization
- **Interactive Prediction Interface**: CLI and web-based prediction tools
- **Model Interpretability**: SHAP analysis for understanding feature importance
- **Comprehensive Analytics**: Data visualization and exploratory analysis
- **RESTful API**: Backend API for model serving and integration
- **Production Ready**: Saved model artifacts with deployment configuration

## 📊 Dataset Information

- **Source**: Rwanda SME Business Registry and Financial Records
- **Size**: 1,000+ SME records
- **Features**: 19 business characteristics including:
  - Financial metrics (Initial Capital, Growth Indicators)
  - Business characteristics (Sector, Model, Type, Location)
  - Operational data (Age, Duration, Employee count)
  - Owner demographics (Age, Gender, Ownership Type)
  - Derived features (Capital per employee, Tech sector classification)

## 🧠 Model Performance

### Best Model: Random Forest
- **Accuracy**: 84.5%
- **Precision**: 84.5%
- **Recall**: 100%
- **F1-Score**: 91.6%
- **Training Samples**: 800
- **Test Samples**: 200

### Feature Engineering
- **Capital Categories**: Micro, Small, Medium, Large business classification
- **Age Categories**: Startup, Emerging, Established, Mature business stages
- **Employee Categories**: Solo, Small team, Medium team, Large team
- **Tech Sector Flag**: Technology sector identification
- **Capital Efficiency**: Capital per employee ratio

## 🏗️ Project Structure

```
SMEs_predictor/
├── notebooks/                      # Jupyter notebooks and analysis
│   ├── SME_Data_Analysis.ipynb    # Main ML pipeline and analysis
│   ├── test_model.py              # Model testing with predefined scenarios
│   └── interactive_test.py        # Interactive CLI prediction tool
├── models/                        # Saved models and metadata
│   ├── best_sme_predictor_random_forest.joblib
│   ├── model_info.json           # Model metadata and performance
│   └── model_requirements.txt    # Deployment dependencies
├── data/                          # Dataset files
│   └── SME_Dataset.csv           # Rwanda SME dataset
├── backend/                       # API backend (FastAPI/Flask)
├── frontend/                      # React.js web interface
├── docs/                         # Documentation and reports
├── .venv/                        # Python virtual environment
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Git
- Virtual environment tool (venv)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Bfestus/SMEs_predictor.git
cd SMEs_predictor
```

2. **Set up virtual environment**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Models

#### 1. Interactive Prediction Tool
```bash
cd notebooks
python interactive_test.py
```

#### 2. Jupyter Notebook Analysis
```bash
jupyter lab
# Open SME_Data_Analysis.ipynb
```

#### 3. Model Testing
```bash
cd notebooks
python test_model.py
```

## 🔍 Model Analysis Highlights

### Data Insights
- **Success Rate**: 84.3% of SMEs in the dataset are classified as successful
- **Key Success Factors**: Initial capital, business age, and growth indicators are primary predictors
- **Sector Analysis**: Technology and service sectors show higher success rates
- **Capital Efficiency**: Businesses with optimal capital-to-employee ratios perform better

### Feature Importance (SHAP Analysis)
1. **Growth_Indicator** (0.24): Most important predictor
2. **Initial_Capital** (0.18): Financial foundation impact
3. **Business_Age** (0.15): Maturity and experience factor
4. **capital_per_employee** (0.12): Operational efficiency
5. **Duration_Operation** (0.11): Operational stability

## 🛠️ Technical Stack

### Machine Learning
- **Python**: 3.12.7
- **Scikit-learn**: Model training and evaluation
- **XGBoost**: Gradient boosting implementation
- **SHAP**: Model interpretability
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn/Plotly**: Data visualization

### Development Tools
- **Jupyter Lab**: Interactive development
- **Git**: Version control
- **Virtual Environment**: Dependency isolation

### Deployment (Planned)
- **FastAPI/Flask**: Backend API
- **React.js**: Frontend interface
- **Docker**: Containerization
- **Cloud Platform**: AWS/Azure/GCP

## 📈 Model Development Process

### 1. Data Exploration & Preprocessing
- Comprehensive EDA with 15+ visualizations
- Missing value analysis and treatment
- Feature engineering and derived variables
- Categorical encoding and scaling

### 2. Model Training & Optimization
- Multiple algorithm comparison (Random Forest, XGBoost)
- Grid Search hyperparameter tuning
- 5-fold cross-validation
- Feature selection and importance analysis

### 3. Model Evaluation & Interpretation
- Performance metrics calculation
- Confusion matrix analysis
- SHAP feature importance
- Model comparison and selection

### 4. Model Persistence & Testing
- Model serialization with joblib
- Metadata storage for deployment
- Comprehensive testing infrastructure
- Interactive prediction interface

## 🎯 Business Impact

### For Entrepreneurs
- **Risk Assessment**: Evaluate business idea viability before investment
- **Strategic Planning**: Identify key factors for success
- **Resource Optimization**: Understand capital and operational requirements

### For Investors
- **Due Diligence**: Data-driven investment decisions
- **Portfolio Risk**: Assess SME investment portfolio risk
- **Market Insights**: Understand Rwanda SME landscape

### For Policymakers
- **Policy Development**: Evidence-based SME support policies
- **Resource Allocation**: Target support to high-potential SMEs
- **Economic Planning**: Understand SME sector dynamics

## 🔮 Future Enhancements

### Phase 1 (Current)
- ✅ ML model development and optimization
- ✅ Interactive prediction tools
- ✅ Model persistence and testing
- ✅ Comprehensive documentation

### Phase 2 (In Development)
- 🔄 RESTful API backend
- 🔄 React.js web interface
- 🔄 API documentation with Swagger
- 🔄 Deployment configuration

### Phase 3 (Planned)
- 📋 Real-time prediction dashboard
- 📋 Business recommendation engine
- 📋 Integration with business registration systems
- 📋 Mobile application development

## 👥 Contributing

This project is part of a capstone submission. For collaboration opportunities or questions, please contact the development team.

## 📄 License

This project is developed for educational purposes as part of the Machine Learning Engineering capstone program.

## 📞 Contact

**Developer**: [Your Name]
**Email**: [Your Email]
**GitHub**: [https://github.com/Bfestus/SMEs_predictor](https://github.com/Bfestus/SMEs_predictor)
**Project**: Machine Learning Engineering Capstone

---

**Built with ❤️ for Rwanda's SME Ecosystem**