"""
Created on Wed Jun  8 17:01:56 2022

@author: Tshepiso

import numpy as np
import joblib
import sys


data = np.loadtxt(sys.stdin)
trained_model = joblib.load('logistic_Regression.pkl')
if(data.shape == (2352 ,)):
    data = [data]
    sys.stdout.write(str(trained_model.predict(data)[0]))
else:    
    for data_point in data:
        sys.stdout.write(str(trained_model.predict([data_point])[0]))
