import pandas as pd

test1 = pd.read_csv("num_test2.csv")
train1 = pd.read_csv("num_train2.csv")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler=StandardScaler()

scaler.fit(test1)
scaled_test_df1= scaler.transform(test1)

scaler.fit(train1)
scaled_df1 = scaler.transform(train1)


X1=scaled_df1[:,0:12]
y1=train1['Price']


x_train1, x_test1, y_train1, y_test1 = train_test_split(    
    X1, y1, test_size=0.15, random_state=3)
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,cross_val_score
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=12,8
from sklearn.externals import joblib
loaded_model = joblib.load("xgb1.joblib.dat")
# print(x_test1[1])
# print(y_test1[2667])

# print(y_pred)

from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def train():
   return render_template('input_seller.html')

@app.route('/result',methods = ['POST', 'GET'])

def result():
   dictt={}
   listt=[]
   dictt={"correct_value":y_test1[2667]}
   if request.method == 'POST':
   		result = request.form
   		for i,j in result.items():
   			if i not in "Namee":
   				listt.append(float(j))
   			else:
   				Name=j
   print("listt= ",listt)
   y_pred = loaded_model.predict(listt)
   print(y_pred)
   dictt={"Name":Name}
   dictt["correct_value"]=y_test1[2667]
   dictt["predicted"]=y_pred
   return render_template("predict.html",result = dictt)



if __name__ == '__main__':
   app.run(port =8081,debug = True)

