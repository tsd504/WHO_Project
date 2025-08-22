# WHO Project - Life Expectancy Prediction

This project focuses on predicting life expectancy using comprehensive health data from the World Health Organisation (WHO). The goal is to develop machine learning models that can accurately estimate life expectancy based on various health indicators, enabling policymakers and healthcare professionals to identify key factors influencing population health outcomes.

## Project Overview

Life expectancy is a critical indicator of population health and development. This project utilises machine learning techniques to analyse WHO health data and predict life expectancy based on demographic, economic, and health-related factors. By understanding the key predictors of life expectancy, we can identify areas for intervention and policy development to improve global health outcomes.

## Project Structure

```
WHO_Project/
├── 1. Harry/               
│   ├── WHO Project - EDA.ipynb           # Comprehensive exploratory data analysis
│   ├── Life Expectancy Predictor Function v1.ipynb # Initial prediction function
│   ├── Test v1.ipynb                    # Model testing and validation
│   └── Life Expectancy Data.csv         # WHO health dataset
├── 2. Rahul/               
│   ├── EDA_Base.ipynb                   # Base exploratory analysis
│   ├── EDA_Limited.ipynb                # Ethical Feature analysis
│   ├── Life Expectancy Predictor Function v1.ipynb # Prediction function
│   ├── limited_model.pkl                # Ethical feature model
│   ├── limited_scaler.pkl               # Ethical feature scaler
│   └── Life Expectancy Data.csv         # WHO health dataset
├── 3. Tom/                
│   ├── PT.ipynb                         # Power transformer analysis
│   ├── Random.ipynb                     # Random model testing
│   ├── Function.ipynb                   # Final prediction function
│   ├── opti_model.pkl                   # Optimised model
│   ├── opti_scaler.pkl                  # Optimised scaler
│   └── Life Expectancy Data.csv         # WHO health dataset
├── Submission_files/         
│   ├── Models.ipynb                     # Complete model analysis
│   ├── EDA.ipynb                        # Final exploratory analysis
│   ├── Function.ipynb                   # Production prediction function
│   └── Life Expectancy Data.csv         # WHO health dataset
└── WHO Dataset - MetaData.ipynb # Dataset documentation and field descriptions
```

## Dataset Overview

The project utilises a comprehensive WHO health dataset containing **2,866 records** with the following key information:

### Geographic and Temporal Data
- **Country**: 193 countries represented
- **Region**: Geographic regions for analysis
- **Year**: Data collection period (2000-2015)

### Health Indicators
- **Life Expectancy**: Target variable (years)
- **Adult Mortality**: Probability of dying between 15-60 years per 1000 population
- **Infant Deaths**: Number of infant deaths per 1000 population
- **Under-five Deaths**: Number of under-five deaths per 1000 population
- **BMI**: Average Body Mass Index of entire population
- **Incidents HIV**: HIV/AIDS deaths per 1000 live births (0-4 years)

### Immunisation Coverage
- **Hepatitis B**: Immunisation coverage among 1-year-olds (%)
- **Measles**: Number of reported cases per 1000 population
- **Polio**: Polio immunisation coverage among 1-year-olds (%)
- **Diphtheria**: DTP3 immunisation coverage among 1-year-olds (%)

### Economic and Social Factors
- **GDP per capita**: Gross Domestic Product per capita (USD)
- **Population**: Country population (millions)
- **Schooling**: Number of years of schooling
- **Alcohol Consumption**: Per capita alcohol consumption (15+ years, litres)
- **Thinness**: Prevalence of thinness among children (5-9 and 10-19 years)

### Target Variable
- **Life Expectancy**: Continuous variable representing average life expectancy in years
- **Range**: 44.3 to 89.0 years
- **Distribution**: Varies significantly by country and development status

## Key Findings

### Data Characteristics
- **Global Coverage**: 193 countries across multiple regions
- **Time Span**: 15 years of data (2000-2015)
- **Data Quality**: Some missing values in immunisation and economic indicators
- **Regional Variation**: Significant differences in life expectancy across regions

### Important Features
1. **Adult Mortality**: Strongest negative correlation with life expectancy
2. **GDP per capita**: Positive correlation with life expectancy
3. **Schooling**: Education level strongly associated with longer life
4. **Immunisation Coverage**: Higher coverage linked to better outcomes
5. **Infant/Child Mortality**: Critical indicators of population health

### Regional Patterns
- **Developed Countries**: Higher life expectancy, better healthcare access
- **Developing Countries**: Lower life expectancy, greater health challenges
- **Geographic Variations**: Regional clustering of health outcomes

## Methodology

### 1. Exploratory Data Analysis (EDA)
- **Data Quality Assessment**: Identified missing values and data inconsistencies
- **Feature Analysis**: Explored relationships between variables and life expectancy
- **Visualisation**: Created comprehensive charts and correlation matrices
- **Statistical Insights**: Analysed distributions and regional patterns

### 2. Feature Engineering
- **Categorical Encoding**: One-hot encoding for region variables
- **Feature Creation**: 
  - Log transformation of GDP per capita
  - Average immunisation score (Polio, Diphtheria, Hepatitis B)
- **Data Cleaning**: Handled missing values and standardised formats
- **Multicollinearity Analysis**: Identified and addressed redundant features

### 3. Machine Learning Models
- **Linear Regression**: Primary model with Lasso-based feature selection
- **Lasso Regression**: Used for feature selection only (not final model)
- **Feature Selection**: Lasso identified 15 important features, then standard linear regression built on selected features

### 4. Model Evaluation
- **Performance Metrics**: R², Mean Absolute Error, Root Mean Square Error
- **Feature Importance**: Understanding which variables most influence life expectancy
- **Cross-validation**: Ensured robust model evaluation

## Results and Insights

### Model Performance
- **Best Model**: Linear Regression with Lasso-based feature selection
- **R² Score**: 0.984 on training set, 0.983 on test set
- **Business Impact**: Clear understanding of life expectancy drivers



## Technical Implementation

### Technologies Used
- **Python**: Primary programming language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning algorithms
- **Matplotlib/Seaborn**: Data visualisation
- **Statsmodels**: Statistical modeling and diagnostics

### Key Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV  # For feature selection only
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm  # For final linear regression model
```



## Future Enhancements

### Model Improvements
- **Deep Learning**: Neural networks for complex pattern recognition
- **Ensemble Methods**: Combining multiple models for better performance


### Feature Engineering
- **External Data**: Integration with additional health indicators
- **Interaction Terms**: Complex feature combinations
- **Temporal Features**: Year-over-year change analysis

### Business Applications
- **Policy Planning**: Evidence-based healthcare interventions
- **Resource Allocation**: Targeting healthcare investments
- **International Comparisons**: Benchmarking health systems

---

*This project showcases the complete data science pipeline from exploratory analysis to actionable health insights, demonstrating how machine learning can drive evidence-based public health policy and interventions.*
