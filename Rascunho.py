# -*- coding: utf-8 -*-
"""
Created on Tue May 27 15:03:04 2019

@author: Gabriel
"""

import pandas_datareader.data as web
import pandas as pd
import datetime
import numpy as np
import math
from matplotlib import style
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy import stats

######################################
############ Parameter ###############
######################################

start = datetime.datetime(2017, 1, 3)
end = datetime.datetime(2017, 10, 4)
num_simulations = 1000;
predicted_days = 250;

######################################
########## Acquiring Data ############
######################################

WEG_d = pd.read_excel(r'C:\Users\Gabriel\Desktop\VerificacaoUFCG\Weg_Stock_Data.xlsx',parse_dates=True,index_col=0)
WEG_p = WEG_d.iloc[:,1]
prices = WEG_p
returns = prices.pct_change()


######################################
####### Monte Carlo Simulation #######
######################################
"""    
last_price = prices[-1]
        
simulation_df = pd.DataFrame()

#Create Each Simulation as a Column in df
for x in range(num_simulations):
    count = 0
    daily_vol = returns.std()
            
    price_series = []
            
    #Append Start Value
    price = last_price * (1 + np.random.normal(0, daily_vol))
    price_series.append(price)
            
    #Series for Preditcted Days
    for i in range(predicted_days):
        if count == 251:
           break
        price = price_series[count] * (1 + np.random.normal(0, daily_vol))
        price_series.append(price)
        count += 1
        
    simulation_df[x] = price_series
"""          
######################################
######### Brownian Motion ############
###################################### 
 
last_price = prices[-1]
 
#Note we are assuming drift here
simulation_df = pd.DataFrame()
        
#Create Each Simulation as a Column in df
for x in range(num_simulations):
            
   #Inputs
   count = 0
   avg_daily_ret = returns.mean()
   variance = returns.var()
            
   daily_vol = returns.std()
   daily_drift = avg_daily_ret - (variance/2)
   drift = daily_drift - 0.5 * daily_vol ** 2
            
   #Append Start Value    
   prices = []
            
   shock = drift + daily_vol * np.random.normal()
   last_price * math.exp(shock)
   prices.append(last_price)
            
   for i in range(predicted_days):
      if count == 251:
         break
      shock = drift + daily_vol * np.random.normal()
      price = prices[count] * math.exp(shock)
      prices.append(price)
                
        
      count += 1
   simulation_df[x] = prices   

######################################
############# Line PLOT ##############
######################################
   
#last_price = prices[-1]
fig = plt.figure()
style.use('bmh')
        
title = "Monte Carlo Simulation: " + str(predicted_days) + " Days"
plt.plot(simulation_df)
fig.suptitle(title,fontsize=18, fontweight='bold')
plt.xlabel('Day')
plt.ylabel('Price ($USD)')
plt.grid(True,color='grey')
plt.axhline(y=last_price, color='r', linestyle='-')
plt.show()

######################################
############# Histogram 1 ############
######################################

ser = simulation_df.iloc[-1, :]
x = ser
mu = ser.mean()
sigma = ser.std()
        
num_bins = 20
# the histogram of the data
n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='blue', alpha=0.5)
         
# add a 'best fit' line
y = mlab.normpdf(bins, mu, sigma)
plt.plot(bins, y, 'r--')
plt.xlabel('Price')
plt.ylabel('Probability')
plt.title(r'Histogram of Speculated Stock Prices', fontsize=18, fontweight='bold')
 
# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.show()
   

######################################
############# Histogram 2 ############
######################################


last_price = prices[-1]
        
price_array = simulation_df.iloc[-1, :]
price_array = sorted(price_array, key=int)  
var =  np.percentile(price_array, 1)
        
val_at_risk = last_price - var
print("Value at Risk: ", val_at_risk)
        
#Histogram
fit = stats.norm.pdf(price_array, np.mean(price_array), np.std(price_array))
plt.plot(price_array,fit,'-o')
plt.hist(price_array,normed=True)
plt.xlabel('Price')
plt.ylabel('Probability')
plt.title(r'Histogram of Speculated Stock Prices', fontsize=18, fontweight='bold')
plt.axvline(x=var, color='r', linestyle='--', label='Price at Confidence Interval: ' + str(round(var, 2)))
plt.axvline(x=last_price, color='k', linestyle='--', label = 'Current Stock Price: ' + str(round(last_price, 2)))
plt.legend(loc="upper right")
plt.show()

print( '#----------------------Descriptive Stats-------------------#')
price_array = simulation_df.iloc[-1, :]
print (price_array.describe())
print ('\n')

print ('#-----------------Calculate Probabilities-------------------------#')
prob1 = (len([i for i in price_array if i > 18.2])/len(price_array))*100
print ("Probability price is higher than initial value: ",prob1,"%") 


