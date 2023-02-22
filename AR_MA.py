import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from matplotlib import pyplot
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
import math
from statsmodels.tsa.arima.model import ARIMA


###########################################################################
df = pd.read_csv('EE627A_HW2_Q2.csv')
rolling_mean = df.rolling(10).mean()
rolling_std = df.rolling(10).std()
plt.title('HW2 Q2 Data')
plt.xlabel("X")
plt.ylabel("Y")
plt.plot(df, color = 'blue')
plt.plot(rolling_mean, color = 'red')
plt.plot(rolling_std, color = 'black')
plt.show()
#this time series is therefore stationary since mean and standard deviation
#stay are normal over the entire time series. 
plot_acf(df)
pyplot.show()
plot_pacf(df)
pyplot.show()

###########################################################################
df = read_csv('EE627A_HW2_Q3.csv')
plt.plot(df['Data'])
plt.title('HW2 Q3 Data')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
#this is not a time series, so we need to convert it to one before applying
#any of our models
df_new = pd.DataFrame({'Data':np.zeros(len(df))})
for ind in df.index:
    if ind >= 1:
        df_new['Data'][ind] = df['Data'][ind] - df['Data'][ind-1]
# #This could now be considered a stationary time series
plt.plot(df_new)
plt.show()
plot_acf(df_new)
pyplot.show()
plot_pacf(df_new)
pyplot.show()
# #none of our models fit, so will need to use ARIMA model
# #The "integrated" term in our ARIMA model will be set to one in this case
# #since we did one difference to convert to a time series
model = ARIMA(df_new.values, order = (4, 0, 3))
model_fit = model.fit()
residuals = pd.DataFrame(model_fit.resid)
plot_acf(residuals)
pyplot.show()
plot_pacf(residuals)
pyplot.show()
###########################################################################