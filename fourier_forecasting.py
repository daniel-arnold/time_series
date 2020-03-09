# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:12:28 2020

@author: danie
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fft

class Fourier_Forecast:
    
    def __init__(self, data, d, k):
        self.data = data
        self.d = d #timestep
        self.k = k #forecast horizon
        self.n = data.size
        self.signal = np.zeros(len(data) + k)
        
    def fit(self, num_freqs):
        
        #find freqs for largest PSD components
        
        #compute fourier transform
        x_fft = scipy.fft.fft(self.data)
        
        #compute power spectral density
        x_psd = np.abs(x_fft)**2
        
        #compute fourier frequencies
        freqs = scipy.fft.fftfreq(self.n, self.d)
        
        #find largest PSD components
        idxs_psd_freqs = list(range(len(freqs)))
        idxs_psd_freqs.sort(key = lambda i: np.log10(x_psd[i]), reverse=True)
        
        #reconstitute the timeseries with the fourier frequencies
        t = np.arange(0, self.n + self.k)
        for j in idxs_psd_freqs[:1 + 2*num_freqs]:
            magnitude = np.abs(x_fft[j])/self.n
            phase_angle = np.angle(x_fft[j])
            self.signal += magnitude * np.cos(2 * np.pi * freqs[j] * self.d * t + phase_angle)
            
        return self.signal
        
        
############################################################################## 
     
if __name__ == "__main__":
    
    #load the data
    df = pd.read_excel("Superstore.xls")
    #df.loc accesses a group of rows and columns by labels or a boolean array
    furniture = df.loc[df.Category == "Furniture"]
    
    #pull out just order date and sales
    furniture = furniture[['Order Date', 'Sales']].sort_values('Order Date')
    
    #multiple orders are placed on the same day, so we will need to sum them up
    #grouping by the order date, sum up the sales and reset the index
    furniture = furniture.groupby('Order Date')['Sales'].sum().reset_index()
    
    #set the index as the order date
    furniture = furniture.set_index('Order Date')
    
    #resample the data into months
    y = furniture['Sales'].resample('MS').mean()
    y.plot()
    plt.ylabel('Sales')
    plt.title('Average Monthly Sales')
    plt.show()
    
    #split the data into training and testing sets
    y_train = y.values[0:36]
    y_test = y.values[36:48]
    
    #predict and forecast the timeseries using fourier decomposition
    d = 1/12
    k = 12
    num_freqs = 6
    fourier = Fourier_Forecast(y_train, d, k)
    signal = fourier.fit(num_freqs)
    
    #plot results
    fig, ax = plt.subplots(1,1, figsize=(8,4))
    ax.plot(y.values,'b',label='true')
    ax.plot(signal,'r',label='prediction')
    ax.axvline(x=36, c='black', lw=3)
    ax.set_xlabel('month')
    ax.set_ylabel('sales')
    ax.set_title('Average Monthly Sales Forecast, fourier frequencies: ' + str(num_freqs))
    plt.legend()
    plt.show()