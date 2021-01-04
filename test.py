

import random
import pandas as pd
from statsmodels.tsa.forecasting.stl import STLForecast
from statsmodels.tsa.arima.model import ARIMA


def predictTemperature(startDate, endDate, temperature, n):
    """
    This function predict temperature data
    """

    # Here we create proper dates from the string provided
    proper_start_date = pd.to_datetime(startDate)
    # An additional day is needed to get to the end of the day
    proper_end_date = pd.to_datetime(endDate) + pd.Timedelta("1day")

    # Now we create a time index and fit the entire data into a dataframe for better handling
    times = pd.date_range(start=proper_start_date, end=proper_end_date, freq='H')[:-1]
    data = pd.DataFrame(index=times, data=temperature, columns=['temp'])

    """Given that we face a time series problem, we select a time series specific forecasting model which can cope
    with potential (and likely) seasonality. Namely we select an ARIMA combined with STL decompose.
    This model subtracts first the seasonality, which it infers it on its own. Afterwards an ARIMA is fitted on the
    stationary series. Normally we would test for stationarity properly through a Dickey-Fuller test, which is given
    time not possible. In order to make stationarity more likely without testing for it, we take the first difference.
    For p and q we assume 1, which normally would also had to be checked through PCAFs and ACFs"""

    data.index.freq = data.index.inferred_freq
    stlf = STLForecast(data, ARIMA, model_kwargs=dict(order=(1, 1, 1), trend="c"))
    stlf_res = stlf.fit()
    forecasts = stlf_res.forecast(24*n).values
    return forecasts




startDate = '2013-05-01'
endDate = '2013-05-03'

temperature = []
for i in range(72):
    temperature.append(random.uniform(20, 30))

n = 2

a = predictTemperature(startDate, endDate, temperature, n)

