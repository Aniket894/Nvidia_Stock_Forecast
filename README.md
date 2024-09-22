# Amazon Stock Forecasting - Detailed Documentation

## Overview
This project involves forecasting Amazon stock prices from **July 1, 2023** to **August 28, 2024** using various machine learning, time series, and deep learning models. The goal is to predict the stock prices for the next 30 days and 90 days based on the **closing price** of the stock and, in some cases, additional features. The project is divided into four notebooks, each exploring different forecasting methods and models.

The notebooks include:
1. Univariate time series forecasting using ARIMA and SARIMA models.
2. Univariate machine learning models like Linear Regression, Random Forest Regressor, Support Vector Regression, and XGBoost.
3. Univariate deep learning forecasting using Long Short-Term Memory (LSTM).
4. Multivariate forecasting using LSTM with multiple input features.

---

## 1. **Nvidia Univariate Stock Forecasting - ARIMA and SARIMA.ipynb**

### Objective:
The primary goal of this notebook is to forecast Amazon stock prices for the next 30 and 90 days using **univariate time series models**. The models used here rely only on the "close" price of Amazon stock, making it a single-variable (univariate) forecasting problem.

### Data:
- **Target Variable**: `close` (Amazon's closing stock price)
- **Date Range**: July 1, 2023 - August 28, 2024

### Models Used:
- **ARIMA** (AutoRegressive Integrated Moving Average):
  - ARIMA is a classic statistical model used for forecasting time series data. It is capable of capturing linear dependencies in the data.
 
    
  - Forecast for the **next 30 days**.
 
  - ![download - 2024-09-22T123058 319](https://github.com/user-attachments/assets/88ee5b69-ecc2-45ed-9654-b5c54d52876b)


  - Forecast for the **next 90 days**.
 
  - ![download - 2024-09-22T123149 040](https://github.com/user-attachments/assets/8505bfdc-0263-4678-b7f8-454f1ee6c533)


  
- **SARIMA** (Seasonal ARIMA):
  - SARIMA extends ARIMA by incorporating seasonality, enabling it to capture periodic fluctuations in stock prices over time.

    
  - Forecast for the **next 30 days**.
 
  - ![download - 2024-09-22T123110 070](https://github.com/user-attachments/assets/d9077c95-8e59-4716-9e87-39e8c001456f)


    
  - Forecast for the **next 90 days**.
  -
  - ![download - 2024-09-22T123200 830](https://github.com/user-attachments/assets/01b41a4c-e810-4c59-bd24-bd387667b4ae)

### Process:
- **Data Preprocessing**: The dataset was cleaned and checked for missing values. A time series was constructed based on the "close" price.
- **Model Tuning**: Both ARIMA and SARIMA models were tuned using grid search to select the optimal p, d, q parameters (for ARIMA) and seasonal components (for SARIMA).
- **Evaluation Metrics**: 
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

### Output:
- **ARIMA Results**: Provided forecasted stock prices for the next 30 and 90 days based solely on past close prices.
- **SARIMA Results**: Extended ARIMA's capability by considering potential seasonal effects, giving an alternative forecast for the next 30 and 90 days.

---

## 2. **Nvidia Univariate Stock Forecasting with Machine Learning Models.ipynb**

### Objective:
This notebook employs various **machine learning models** to forecast Amazon's stock price using the closing price for the next 30 and 90 days. Unlike time series models, these machine learning models aim to capture complex relationships in the data through regression techniques.

### Data:
- **Target Variable**: `close` (Amazon's closing stock price)
- **Date Range**: July 1, 2023 - August 28, 2024

### Models Used:
1. **Linear Regression**:
   - Simple regression model that fits a linear equation to the data.
   
   - Forecast for the **next 30 days**.
  
   - ![download (97)](https://github.com/user-attachments/assets/4cf0d9b0-f6fa-41ed-8b85-31110da7233d)
  
   - ![download (98)](https://github.com/user-attachments/assets/9cbe1af9-3b5a-49a2-b703-637aa0d293be)


   - Forecast for the **next 90 days**.
  
   - ![download (99)](https://github.com/user-attachments/assets/aa213362-9d17-4833-a4e3-38489de20416)
  
   - ![download (100)](https://github.com/user-attachments/assets/b21ae6ec-31a6-4b72-b77b-3cb1248e8d60)



2. **Random Forest Regressor**:
   - A tree-based ensemble learning model that reduces variance and improves prediction accuracy.
     
   - Forecast for the **next 30 days**.

    - ![download - 2024-09-22T122001 661](https://github.com/user-attachments/assets/37f88f16-ad7f-40e1-80bd-84b42b96031e)
  
    - ![download - 2024-09-22T122014 678](https://github.com/user-attachments/assets/628b5043-a412-42a4-82bb-4ade0960fda8)
     
   - Forecast for the **next 90 days**.
  
   - ![download - 2024-09-22T122029 078](https://github.com/user-attachments/assets/11a4ce53-af25-4f2a-84b4-ea0f9e31abc3)
  
   - ![download - 2024-09-22T122042 292](https://github.com/user-attachments/assets/bdcae708-a194-4e4e-8f68-bd5bd94f0103)



4. **Support Vector Regression (SVR)**:
   - SVR fits a hyperplane to the data in a high-dimensional space, useful for non-linear relationships.

     
   - Forecast for the **next 30 days**.
     
   -  ![download - 2024-09-22T122104 756](https://github.com/user-attachments/assets/0125a7a4-9c0c-489c-bf04-a64f039464fd)
  
   -  ![download - 2024-09-22T122114 483](https://github.com/user-attachments/assets/5b480fea-04a0-497d-9114-fc76a298fc63)


   - Forecast for the **next 90 days**.
  
   - ![download - 2024-09-22T122132 831](https://github.com/user-attachments/assets/8aacfd64-5e61-4d6b-a740-76fba50036ab)
  
   - ![download - 2024-09-22T122144 213](https://github.com/user-attachments/assets/b875441d-f8f1-45c4-8587-69fa817a2df6)



5. **XGBoost Regressor**:
   - A highly efficient and accurate boosting algorithm for regression.

     
   - Forecast for the **next 30 days**.
  
   - ![download - 2024-09-22T122203 530](https://github.com/user-attachments/assets/6b47e983-a4dc-4358-bec5-95369250b250)
  
   - ![download - 2024-09-22T122215 513](https://github.com/user-attachments/assets/919f0402-18f6-4b61-81ca-dc72ff5fa063)


   - Forecast for the **next 90 days**.
  
   - ![download - 2024-09-22T122232 599](https://github.com/user-attachments/assets/18372586-d5d8-4c79-8d0a-5599ad747c68)
  
   - ![download - 2024-09-22T122246 073](https://github.com/user-attachments/assets/1e31d09a-ebbe-4909-8034-346063986ec6)



### Process:
- **Data Preprocessing**: Similar to the first notebook, the dataset was cleaned, and feature scaling was applied to ensure models perform optimally.
- **Model Training and Tuning**: Each model was trained using past "close" prices, and hyperparameter tuning was performed to achieve the best performance.
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

### Output:
- **Linear Regression**: Basic linear model provided a straightforward forecast for the next 30 and 90 days.
- **Random Forest**: Improved performance by capturing more complex patterns in stock price movements.
- **SVR**: Captured non-linear relationships, producing smoother predictions.
- **XGBoost**: Delivered the most robust predictions due to its strong regularization and boosting capabilities.

---

## 3. **Nvidia Univariate Stock Forecasting with LSTM.ipynb**

### Objective:
This notebook applies a **deep learning approach** using a Long Short-Term Memory (LSTM) neural network to forecast the Amazon stock price based on historical "close" prices. LSTM is particularly suited for time series data because it can capture long-term dependencies in sequences.

### Data:
- **Target Variable**: `close` (Amazon's closing stock price)
- **Date Range**: July 1, 2023 - August 28, 2024

### Model Used:
- **LSTM (Long Short-Term Memory)**:
  - A type of Recurrent Neural Network (RNN) capable of learning temporal dependencies in time series data.
 
    
  - Forecast for the **next 30 days**.
 
  - ![download - 2024-09-22T131539 157](https://github.com/user-attachments/assets/4dba2be0-21ec-40cc-8aef-e6c35f4a05c1)
 
  - ![download - 2024-09-22T131206 518](https://github.com/user-attachments/assets/b7251917-e03c-48c7-a845-efca43c65a4f)


  - Forecast for the **next 90 days**.
 
  - ![download - 2024-09-22T131712 566](https://github.com/user-attachments/assets/64d1476d-8f9b-45e8-b37b-74903a08f470)
 
  - ![download - 2024-09-22T131619 105](https://github.com/user-attachments/assets/39bc55c4-f372-455b-81b1-9c14da982654)



### Process:
- **Data Preprocessing**: The "close" prices were normalized using Min-Max scaling. A sliding window technique was used to create sequences for training the LSTM model.
- **Model Architecture**:
  - The LSTM model was built with multiple layers, including an LSTM layer, dropout layers to prevent overfitting, and a dense output layer for prediction.
- **Training**:
  - The model was trained on historical data using the Adam optimizer, with the loss function set to Mean Squared Error (MSE).
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

### Output:
- **LSTM Results**: The deep learning model predicted the next 30 and 90 days of Amazon stock prices based on the sequential nature of past "close" prices, with a focus on long-term trends.

---

## 4. **Nvidia Multivariate Stock Forecasting with LSTM.ipynb**

### Objective:
Unlike the previous notebooks, this one incorporates multiple features (multivariate) for stock price forecasting. The objective is to leverage additional stock-related variables along with the "close" price to improve the forecasting performance using an LSTM model.

### Data:
- **Target Variable**: `close` (Amazon's closing stock price)
- **Additional Features**: Other stock-related variables such as `volume`, `open`, `high`, `low`, etc.
- **Date Range**: July 1, 2023 - August 28, 2024

### Model Used:
- **LSTM (Long Short-Term Memory)**:
  - A deep learning model that can handle multivariate inputs, capturing temporal dependencies across multiple features.
  - Forecast for the **next 30 days**.
 
  - ![download (93)](https://github.com/user-attachments/assets/9e8b6dbf-9701-4d12-91ed-3d68752097c7)
 
  - ![download (94)](https://github.com/user-attachments/assets/7aa80f82-c8dd-48a5-9ad6-ee78f19b8802)


  - Forecast for the **next 90 days**.
 
  - ![download (95)](https://github.com/user-attachments/assets/7f7ac4a5-9aa6-4363-aad8-fbcf54502bc9)
 
  - ![download (96)](https://github.com/user-attachments/assets/c2d9a315-9aea-4643-9e55-879f4069f188)



### Process:
- **Data Preprocessing**: Multiple columns (including "close", "volume", "open", "high", and "low") were normalized. A sliding window technique was applied to create multivariate sequences for training the model.
- **Model Architecture**:
  - Similar to the univariate LSTM model, but adapted to handle multivariate inputs, with additional layers and neurons to accommodate the larger feature set.
- **Training**:
  - The LSTM model was trained using the Adam optimizer, and hyperparameter tuning was done to optimize performance.
- **Evaluation Metrics**:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)

### Output:
- **LSTM Results**: The model used multiple features to predict the next 30 and 90 days of Amazon stock prices. The multivariate approach helped improve prediction accuracy by incorporating additional information beyond the "close" price.

---

## Conclusion
This project demonstrates multiple approaches to forecasting Amazon stock prices using univariate and multivariate models. Traditional time series models (ARIMA, SARIMA), machine learning algorithms (Linear Regression, Random Forest, SVR, XGBoost), and deep learning models (LSTM) were applied to the problem, each providing unique insights into stock price movements. By comparing these models, one can choose the best approach depending on the desired forecast horizon and data complexity.
