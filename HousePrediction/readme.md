# House Price Prediction with Machine Learning and Data Visualization

## Contributors
- Muhammad Abbas 23k-0068
- Hamza Sheikh 23k-0060
- Sabeeh 23k-0002

---

## Objective
Develop a Machine learning tool that predicts house prices based on various datasets and ML features like linear regression, feature engineering etc and provides insightful data visualizations to understand housing data trends.

---

## Dataset Description
- **Dataset Name:** Housing.csv  
- **Number of Instances:** Not specified  
- **Number of Attributes:** 12  
- **Attribute Breakdown:**
  - **Categorical Attributes:** *mainroad*, *guestroom*, *basement*, etc.
  - **Numerical Attributes:** *area*, *bedrooms*, *bathrooms*, etc.
- **Missing Values:**
  - Categorical columns were imputed using the most frequent values.  
  - Numerical columns were imputed using the mean.

---

## Methodology

### 1. Data Loading and Preprocessing
- Loaded the housing dataset using Pandas.
- Encoded categorical variables (*yes/no* to *1/0*) and applied scaling on numerical columns.
- Implemented feature engineering, including interactions between numerical features (commented for optimization as accuracy levels were decreasing due to it).

### 2. Statistical Analysis
- Derived basic statistical insights using:
  - Histograms
  - Boxplots
  - Correlation heatmap  
- Performed regression analysis on significant features to identify patterns with the target variable (*price*).

### 3. Machine Learning Model
- **Model Used:** Linear Regression  
- **Pipeline Implementation:**
  - Preprocessing with numerical and categorical transformations.
  - Supervised learning using Linear Regression.  
- **Train-Test Split:** 90% training, 10% testing.  
- **Metrics:**
  - Mean Absolute Error (MAE): *Calculated*  
  - R² Score: *Reported*  
- **Model Export:** Saved as `house_price_model.pkl` for integration with the GUI.

---

## Application Features

### Graphical User Interface (Tkinter Framework)
#### Prediction
- Users can input house features to predict the price. Validation ensures accurate data input.

#### Graphs and Visualizations
- **Histograms:** Feature distributions.  
- **Boxplots:** Identify outliers.  
- **Correlation Heatmaps:** Highlight relationships between features.  
- **Scatter Plots:** Price vs. significant features.

#### Interactivity
- Dynamic navigation.
- Tooltips for input guidance.
- Reset option for input fields.

---

## Data Visualization
Implemented using Matplotlib and Seaborn to analyze continuous and categorical data:
- **Feature-Specific Distributions**: Histograms and boxplots.  
- **Regression Plots**: Significant feature vs. price.

---

## Dependencies and Tools
- **Programming Language:** Python  
- **Libraries:**
  - Pandas  
  - Matplotlib  
  - Seaborn  
  - Scikit-learn  
  - Tkinter  
- **Additional Tools:**
  - OneHotEncoder for categorical data.  
  - StandardScaler for normalization.  

---

## Conclusion
This project successfully combines machine learning, data visualization, and GUI-based interactivity to deliver a comprehensive house price prediction system. By integrating modular design and an intuitive interface, it enables robust data exploration and accurate predictions.

---


