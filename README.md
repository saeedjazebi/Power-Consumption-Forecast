# Power-Consumption-Forecast
## The main objective of this project is to develop a predictive model for power consumption based on time series data.
### A predictive model for power consumption offers several significant advantages across multiple sectors, specifically for Utilities and Grid Operators:
1.	Infrastructure planning - Make informed decisions about when and where to expand capacity
2.	Load balancing and optimal operation - Better distribution of electricity across the grid, reducing strain and preventing outages
3.	Demand forecasting - Accurately predict peak demand periods to optimize generation capacity
4.	Resource optimization - Schedule maintenance during predicted low-demand periods
5.	Renewable integration - Better manage the intermittent nature of renewable energy sources

### The main beneficiaries of such models could be the following:
1.	Utility companies - Improved grid management, reduced costs, better service reliability
2.	Energy-intensive industries (manufacturing, data centers, etc.) - Significant cost savings and operational improvements
3.	Commercial real estate - Better building management and reduced operating expenses
4.	Governments and municipalities - Improved public infrastructure management
5.	Renewable energy providers - Better integration with traditional power sources
6.	End consumers - Lower bills and improved service reliability

## Useful Links
- Link to Jupyter Notebook: https://github.com/saeedjazebi/Power-Consumption-Forecast/blob/main/CapstoneFinal.ipynb
- Data Source: https://www.kaggle.com/datasets/fedesoriano/electric-power-consumption
## Data Description
This project develops a predictive model for power consumption based on time series weather data (temperature, humidity, wind speed, and diffuse flow). The data is collected from Supervisory Control and Data Acquisition System (SCADA) of Amendis which is a public service utility provider and in charge of the distribution of water and electricity since 2002, in Morocco. The dataset is exhaustive in its demonstration of energy consumption of the Tétouan city in Morocco. The city is fed by three main distribution substations, that each serve a particular zone, namely: Quads, Smir, and Boussafou. The objective is to predict energy consumption in advance to inform electricity prices in a dynamic electricity market. This data could also be used to support data-driven decision making.
Data Set Description The data consists of 52,416 observations of energy consumption on 10-minute intervals (6 samples per hour). Every observation is described by 9 feature columns. The data is collected for a full year.
1.	Date Time: Time window of ten minutes (Input Feature)
2.	Temperature: Weather Temperature (Input Feature)
3.	Humidity: Weather Humidity (Input Feature)
4.	Wind Speed: Wind Speed (Input Feature)
5.	General Diffuse Flows (Input Feature)
6.	Diffuse Flows (Input Feature)
7.	Zone 1 Power Consumption (Target Feature)
8.	Zone 2 Power Consumption (Target Feature)
9.	Zone 3 Power Consumption (Target Feature)
## Exploratory Data Analysis (EDA)
Exploratory Data Analysis (EDA) started with importing the required libraries. Read the data from csv files. Explore the data, feature names, and type of the data. The data is all numerical. Only Datetime feature need to be converted to proper format. Explored missing values, but the data set does not have missing values. Checked the statistics of all data fields to achieve a better understanding. Checked the correlation between different variables. Temperature and power consumption have the highest positive correlation. Humidity has a negative correlation with temperature and power consumption. Wind speed increases power consumption generally. DiffuseFlow has generally lower correlation with other parameters, especially it seems to have no correlation with target variables (power consumption). Checked the distribution of data via histograms. No data anomaly observed. Used Pairplot to visualize relationships between numerical columns. Convert the 'Datetime' column to datetime format. Plotted the energy consumption for the three zones on the same plot and compared the time-series behavior between the three zones. Normalized all columns except 'Datetime' to be able to plot all important variables on the same plot. Plotted the Power consumption for each zone with temperature, humidity and windspeed to visualize the trends between the variables. Generally, temperature follows the same trend as power consumption. Looking at the distribution of the data it does not seem that we have a lot of outliers in the data. Using boxplots for power consumption of zone 1, 2, and 3, it looks like zone 3 has some outliers. Since the data is time-series data, if the outliers are removed, they should be imputed by mean, mode, or their next data sample, which might skew the data in other ways, so decided not to remove the outliers for this trial. No duplicate rows were found in the dataset. 

## Modeling
Four different general models are used in this project: 
1.	Predictive Regression 
2.	Time-series Autoregression, ARIMA & SARIMA method 
3.	Facebook Prophet Model 
4.	Deep learning with Neural Networks.

### 1- Predictive Regression: 
Predict power consumption for three different regions based on Temperature, Humidity, WindSpeed, GeneralDiffuseFlows, and DiffuseFlows. For this method, the time component could be dropped from the data.
Used Sequential Feature Selection (SFS) method to identify which parameters are most important and critical to predict power consumption in Zone 1, 2, and 3. The Temperature, Humidity, and WindSpeed seem to be the most critical features. Hence, I used these three features as input variables.
Tested both Ridge and Lasso regressions. Used GridSearchCV to optimize both. Used the root mean square error to assess the performance of the regression models. The Ridge regression is more accurate.
- Lasso Testing RMSE: 0.8825
- Ridge Testing RMSE: 0.8616\
Also, Lasso revealed that for each zone different combinations of features provide the best performance, by selecting those features: 
- Zone 1 ---> Temperature // TemperatureHumidity // Humidity^2 
- Zone 2 ---> Temperature // TemperatureWindSpeed // Humidity^2 
- Zone 3 ---> Temperature^2 // Temperature*WindSpeed


### 2- Time Series Analysis using ARIMA & SARIMA
For the time series, i focused on Zone 1 power prediction only. However, it is easy to extend it to the other zones to predict their power consumption. 
Zone 1 power consumption is first decomposed into Trend, Seasonal, and residual components, with period = 24 hours, which means 144 samples (24 hours * 6 sample per hour).
Applied the AD-Fuller algorithm to the residual component. The Time series is stationary based on the Augmented Dickey-Fuller (ADF) test results. The key indicators showing stationarity are:
1.	ADF Statistic: -35.675345. 
2.	p-value: 0.000000 (essentially zero).
3.	Critical Values: 
o	1%: -3.430.
o	5%: -2.862.
o	10%: -2.567.
<br />The ADF statistic (-35.67) is significantly more negative than even the 1% critical value (-3.43), indicating strong evidence to reject the null hypothesis of a unit root. The extremely low p-value (0.000000) further confirms this conclusion, meaning there's overwhelming statistical evidence that your time series is stationary.
The Trend and Seasonal components are assumed to follow the same behavior. As such, the residual component is modelled with ARIMA and SARIMA models and then the predicted residual is added to the Trend and Seasonal components to reconstruct the temperature time series data. This data is then used to compare against the actual data for accuracy.  
After training the ARIMA and SARIMA models, the results were checked against the real data. 
The forecast() method in ARIMA and SARIMA models which is appropriate for out of sample prediction, often yields relatively constant predictions, especially for longer forecast horizons. This happens because these models have a mathematical tendency to revert to mean overtime. As the forecast horizon increases, predictions gradually converge toward the series mean. Also, these models assume stationarity after differencing. For a stationary series, the best long-term forecast is essentially a constant value. 
The root mean square error and the mean square error for both ARIMA and SARIMA are calculated to compare against other time-series predictive models. 

### 3- Facebook Prophet Model
The training and testing data are prepared for the Prophet model. The model is then trained. made predictions for the test data period. A visual comparison is presented for the 24 hours following the training data. The RMSE is calculated as 2470.
### 4- Artificial Neural Networks
For Neural Networks, two different methods are considered. 1) Using a moving window on the power consumption feature only. In this method, the input is 24 hours historical power consumption, and the output is the next 24-hour power consumption. This would predict the next 24-hour power consumption based on historical data. For this reason, three different models were trained: a) Regular neural network b) A Simple Recurrent Neural Network (RNN), c) A Long Short-Term Memory (LSTM) network. The performance of the models has been compared with respect to the RMSE.  2) Using moving windows of historical 24-hour power consumption and temperature, and predicted 24-hour temperature, as input, and the predicted future 24-hour power consumption as output. Hence, the input has 3 components, and the output has one component.  For this method, only LSTM was used. Lastly, hyperparameter tuning is performed to optimize the model parameters.
Created a function that prepares windows of historical and future data for training and testing. Converted the data into input and output variables X, y with this function. The length of the vectors is 144 (24 hours * 6 sample per hour). The training and testing data sets are then generated. The dimensions of the data are checked so that they are correct and also, we can design an appropriate neural network for them.
### Method 1)
#### a)	Regular ANN:
A regular ANN model is built, with 50 dense hidden layers, and a dense output layer with RELU activation function. After 20 Epochs the RMSE is 9555.
#### b)	simple RNN:
A simple RNN model is built with 50 layers, and RELU activation function, and a dense output layer. The RMSE after 20 epoch is 1363.
#### c)	 LSTM 1: 
A LSTM model is built with 50 layers, RELU activation function, and a dense output layer. The RMSE after 20 epochs is 1352.
The simple LSTM method seems to work better for the purpose of Method 1. Therefore, for method 2 only LSTM model is used.

### Method 2)
In previous sections, it was discovered that the most important variable affecting the power prediction was the temperature. Here we assume that for a practical application, we have a proper prediction of temperature for the next 24 hours. Also, we have temperature measurements for the last 24 hours. We also have measured power consumption in the last 24 hours. Hence, we used those known variables as input and predicted the next 24 hours power consumption. 
The input and output variables have different dimensions; therefore, two different functions are written to produce the required moving window of data. The length of windows is kept at 144 sample points. 
#### a)	LSTM 2
A new LSTM model is trained in this section. The RMSE after 20 epochs is 1378. 
The LSTM seems to be the most promising method to predict power consumption. As such, to make sure that we have a well-tuned model, in the next section hyperparameter tuning is performed to adjust the model. 
#### b)	 LSTM 3: Hyperparameter tuning on LSTM
The hyper parameter tuning is performed for the number of LSTM layers and the learning rate. The rest of the parameters are kept constant. 
The table below compares the performance of all time-series models. The test data’s Root Mean Square Error (RMSE) is compared. One can conclude that the LSTM model with hyperparameter tuning is outperforming the rest. For this reason, the actual temperature is compared against this method visually which quite well fits the real data.  
Fitting 2 folds for each of 12 candidates, totaling 24 fits, the best hyperparameters: 200 layers and learning rate of 0.001.

### Performance Comparison of Time-Series Models
- ARIMA RMSE: 2211	
- SARIMA RMSE: 2234	
- Prophet RMSE: 2470	
- ANN RMSE: 9555
- Simple-RNN RMSE:1363
- LSTM 1 RMSE: 1352	
- LSTM 2 RMSE: 1378	
- LSTM 3 RMSE: 860
			

## Next Steps, Future Work, and Recommendations:
- The time component was not considered as an input variable for our Neural network Models. Considering time as input variable could increase the model’s accuracy. For this reason, the Datetime variable should be broken down into multiple features, such as hour, minute, day of the week, quarter, month, day, year, season, day of the month, day of the year, etc. This will increase the model’s power to more accurately predict the power consumption, because different days of the week or different months of the year have direct effect on the pattern of consumption. For examples, there could be another feature called “holiday”, or a variable that indicates if it is “weekday” or “weekend”.

- For time-series analysis, the models are trained and tested only for Zone 1 power consumption. The same models could be applied to Zone 2 and 3 to test the accuracy.

- Models could be built based on three zones power consumption simultaneously and it may increase models’ accuracy, because more data will be used to train the models.   


## Contact and Further Information
Saeed Jazebi
Email: jazebi@ieee.org 
[LinkedIn](linkedin.com/in/jazebi)
