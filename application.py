from flask import Flask, jsonify, request, render_template
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import pickle

#get and post method
'''
GET Method
Used to: Get data from the server.
Example: When you search something on Google, you're using GET.

How it works:

The data you send (like a search term) is added to the URL.
Example: www.google.com/search?q=chatgpt
Visible?: Yes, the data is visible in the URL.
Safe to use for: Searching, reading, or browsing pages.
Limit: Can only send small amounts of data.

📨 POST Method
Used to: Post (send) data to the server to save or update something.
Example: When you fill out a login form or upload a photo.

How it works:
The data is sent in the body of the request (not shown in the URL).
Visible?: No, data is hidden from the URL.
Safe to use for: Login forms, registration, payments, uploading data.
Limit: Can send larger amounts of data than GET.
'''
#import ridge regressor and standard scaler pickle
ridge_model= pickle.load(open('models/ridge.pkl', 'rb'))
standarS=pickle.load(open('models/scaler.pkl', 'rb'))

application=Flask(__name__)
app=application 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictData', methods=['GET' , 'POST'])
def predict_data():
    if request.method=='POST':
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))
        
        new_data_scaled=standarS.transform([[Temperature, RH,Ws, Rain , FFMC, DMC, ISI, Classes, Region]])
        result=ridge_model.predict(new_data_scaled)
        
        return render_template('home.html', results=result[0])
    
    else:
        return render_template('home.html')

if __name__=='__main__':
    app.run(host='0.0.0.0')