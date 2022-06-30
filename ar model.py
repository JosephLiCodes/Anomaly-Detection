from sqlite3 import Time
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_model import ARIMA
from datetime import datetime
from time import time
import calendar

def parser(s):
    return datetime.strptime(s, '%m/%d/%Y %H:%M:%S')
print(parser("05/15/2022 11:00:00"))
def timestr_to_secs(timestr):
    fmt = '%m/%d/%Y %H:%M:%S'
    time_struct = datetime.strptime(timestr, fmt)
    secs = calendar.timegm(time_struct)
    return secs


data = pd.read_csv('RxTx.csv', parse_dates=[6], index_col=6, date_parser = parser)
tx = data[data['Tags'].str.contains("subcomponent=NSX-Edge-1-172-27-236-8")]
tx = tx[tx["Label"].str.contains("SR-vmca.iface.cross-vpc-1i.tx_packets")]
tx = tx["Point"]
#print(tx)
packets = pd.read_csv('RxTx.csv',usecols=['Point'])
time = pd.read_csv('RxTx.csv', usecols=['Time'])
# changedtime = time.apply(timestr_to_secs, axis = 1)
# print(changedtime)
pyplot.plot(tx)
train = tx[0:len(tx)-400]
test = tx[len(tx)-400:len(tx)]
pyplot.show()
     
model = AutoReg(train, lags=16).fit()
#model1 = ARIMA(train, order = (0,1)).fit()


#print(model.summary())
pred = model.predict(start = len(tx)-400,end = len(tx),dynamic = False)
from matplotlib import pyplot
pyplot.plot(pred, color = 'green')
pyplot.show()
#print(pred)