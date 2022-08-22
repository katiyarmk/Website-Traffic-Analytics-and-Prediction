
<h1> Introduction</h1>
The goal of this notebook is not to do the best model for each Time series. It is just a comparison of few models when you have one Time Series. The presentation present a different approaches to forecast a Time Series.In this notebook we will be using web traffic data from kaggle.
<h4> Dataset can be downloaded from here <a href="[url](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting/data)">Dataset</a>
The plan of the notebook is:
<ol>
<li>Importation & Data Cleaning</li>
<li>Aggregation & Visualization</li>
<li>Machine Learning Approach</li>
<li> Basic Model Approach</li>
<li> ARIMA approach (Autoregressive Integrated Moving Average)</li>
<li>(FB) Prophet Approach </li>
<li>Keras Starter </li>
<li> Comparaison & Conclusion </li>
</ol>
<ol>
<li><h2>Importation & Data Cleaning: </h2></li> 
In this first part we will choose the Time Series to work in the others parts. The idea is to find a Time Serie who could be interesting to work with. So in the data we can find 145K Time Series. We will Find a good Time Series to introduce four approaches! So the first step is to import few libraries and the data. The four approaches are Basic Approach / ML Approach / GAM Approach / ARIMA Approach.
Dataset looks like this

<img width="623" alt="image" src="https://user-images.githubusercontent.com/111698968/185917633-8b1772e8-636f-4b01-96d1-454bfcfc208d.png">

This part allowed us to prepare our data. We had created new features that we use in the next steps. Days, Months, Years are interesting to forecast with a Machine Learning Approach or to do an analysis. If you have another idea to improve this first part: Fork this notebook and improve it or share your idea in the comments.

![image](https://user-images.githubusercontent.com/111698968/185917913-29a4cde9-6c5e-4790-9690-09a600d47300.png)


![image](https://user-images.githubusercontent.com/111698968/185918011-1d601365-c620-47a4-a61b-3842c1a1b4cb.png)
This heatmap show us in average the web traffic by weekdays cross the months. In our data we can see there are less activity in Friday and Saturday for December and November. And the biggest traffic is on the period Monday - Wednesday. It is possible to do Statistics Test to check if our intuition is ok. But You have a lot of works !
![image](https://user-images.githubusercontent.com/111698968/185918257-e3536db2-ae39-4faf-8025-ba6c5c31fda3.png)

<li><h2>III. ML Approach</h2></li>


The first approach introduces is the Machine Learning Approach. We will use just a AdaBoostRegressor but you can try with other models if you want to find the best model. I tried with a linear model as like Ridge but ADA model is better. I will be interesting to check if GB or XGB can bit ADA. It is possible to do a Neural Network approach too. But this approach will be done if the kagglers want more !!
Now our Dataset looks like this
<img width="633" alt="image" src="https://user-images.githubusercontent.com/111698968/185918810-f7d537fa-7dd5-4ef5-9d1b-c3ba48123013.png">

After Creating Several Lags
<img width="621" alt="image" src="https://user-images.githubusercontent.com/111698968/185918919-ab9330c2-b59e-4aa8-8680-54e093685a5f.png">

Then We did train test split
<img width="622" alt="image" src="https://user-images.githubusercontent.com/111698968/185919138-17de9f52-7978-49cb-b636-51e36948243f.png">

![image](https://user-images.githubusercontent.com/111698968/185919288-54d477bd-17ca-4ee2-adcb-8b2dfb4fcc64.png)

![image](https://user-images.githubusercontent.com/111698968/185919339-bd501bab-a799-4ee1-a3b5-d7f53422b311.png)


Finshed for the first approach ! The ML method requires a lot of work ! You need to create the features, the data to collect the prediction, optimisation etc… This method done a good results when there are a weekly pattern identified or a monthly pattern but we need more data.


<li><h2>IV. Basic Approach</h2></li>


For this model We will use a simple model with the average of the activity by weekdays. In general rules the simplest things give good results !

Prediction By Days
<img width="623" alt="image" src="https://user-images.githubusercontent.com/111698968/185919535-05fd5937-7a10-4a37-8eaa-7d556347e551.png">

<h3>Prediction With Basic Model</h3>
<img width="623" alt="image" src="https://user-images.githubusercontent.com/111698968/185919891-377e47f9-ecfa-402b-ae97-e96298d4e901.png">

No optimisation ! No choice between linear, Bagging, boosting or others ! Just with an average by week days and we have a result ! Fast and easily !
<h2>V. ARIMA</h2>
This part is inspired by: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/ Very goodjob with the ARIMA models ! It is more simple when we have directly a stationary Time series. It is not our case…

We will use the Dickey-Fuller Test. More informations here: https://en.wikipedia.org/wiki/Dickey%E2%80%93Fuller_test

<img width="628" alt="image" src="https://user-images.githubusercontent.com/111698968/185920178-6c89f714-c314-45e3-8360-e5ad110c46c3.png">
<img width="628" alt="image" src="https://user-images.githubusercontent.com/111698968/185920392-c173e329-ee55-4a3e-a64d-6eff3fbfbd2c.png">


Our Time Series is stationary ! it is a good news ! We can to apply the ARIMA Model without transformations

![image](https://user-images.githubusercontent.com/111698968/185920519-3140ca10-8f7c-4ee0-bf15-144dbedcdcba.png)


Good job ! We have a Time Series Stationary ! We can apply our ARIMA Model !!!

We expose the naive decomposition of our time series (More sophisticated methods should be preferred). They are several ways to decompose a time series but in our example we take a simple decomposition on three parts. The additive model is Y[t] = T[t] + S[t] + e[t] The multiplicative model is Y[t] = T[t] x S[t] x e[t] with:

T[t]: Trend

S[t]: Seasonality

e[t]: Residual

An additive model is linear where changes over time are consistently made by the same amount. A linear trend is a straight line. A linear seasonality has the same frequency (width of cycles) and amplitude (height of cycles). A multiplicative model is nonlinear, such as quadratic or exponential. Changes increase or decrease over time. A nonlinear trend is a curved line.A non-linear seasonality has an increasing or decreasing frequency and/or amplitude over time. In ou example we can see it is not a linear model. So it is the reason why we use a multiplicative model.

<li><h2>VI. Prophet</h2></li>
Prophet is a forecasting tool availaible in python and R. This tool was created by Facebook. More information on the library here: https://research.fb.com/prophet-forecasting-at-scale/

Compared to the two methods this one will be faster. We can forecast a time series with few lines. In our case we will do a forecast and a display the trend of activity on the period and for a week.
INFO:fbprophet.forecaster:Disabling yearly seasonality. Run prophet with yearly_seasonality=True to override this.
INFO:fbprophet.forecaster:Disabling daily seasonality. Run prophet with daily_seasonality=True to override this.
![image](https://user-images.githubusercontent.com/111698968/185920942-c26adaf8-eea0-4cb9-bd0d-f1ee18c3e9d0.png)

<li><h3>Forecast Plot</h3></li>
 <img width="628" alt="image" src="https://user-images.githubusercontent.com/111698968/185923251-f2815d98-9cc2-4c04-9ed6-0e87d342e487.png">

<li><h2>VI. Keras Starter</h2></li>
In this part we will use Keras without optimisation to forecast. It is just a very simple code to begin with Keras and a Time Series. For our example we will try just with one layer and 8 Neurons.
<img width="628" alt="image" src="https://user-images.githubusercontent.com/111698968/185921167-c1abbdc6-f988-45bd-99a4-f42b144f66b6.png">


