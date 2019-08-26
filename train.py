import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
train = pd.read_csv('num_train.csv')
test = pd.read_csv('num_test.csv')

train1 = pd.read_csv('num_train2.csv')
test1 = pd.read_csv('num_test2.csv')

scaler=StandardScaler()
scaler.fit(train)
scaled_df=scaler.transform(train)
scaler.fit(test)
scaled_test_df=scaler.transform(test)

scaler.fit(train1)
scaled_df1 = scaler.transform(train1)
scaler.fit(test1)
scaled_test_df1= scaler.transform(test1)

# Convert to numpy - Classification
X=scaled_df[:,0:11]
y=train['Price']

X1=scaled_df1[:,0:12]
y1=train1['Price']

# Create train/test
x_train, x_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.15, random_state=3)

x_train1, x_test1, y_train1, y_test1 = train_test_split(    
    X1, y1, test_size=0.15, random_state=3)
print(X.shape)
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV,cross_val_score
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=12,8

def modelfit(alg, dtrain,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=y_train1)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], 
                          nfold=cv_folds,
            metrics='rmse', early_stopping_rounds=early_stopping_rounds)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(x_train1,y_train1,eval_metric='rmse')
        
    #Predict training set:
    dtrain_predictions = alg.predict(x_train1)
    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
    #Print model report:
    print ("\nModel Report")
    #print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain['Price'].values, 
    #dtrain_predictions))
    print ("RMSE Score (Train): %f" % metrics.mean_squared_error(y_train1, 
                                                                 dtrain_predictions))
                    
    feat_imp = pd.Series(alg._Booster.get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    return alg
xgb1 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=890,
 max_depth=5,
 objective='reg:squarederror',
 seed=27)

xgb3 = XGBRegressor(
 learning_rate =0.1,
 n_estimators=500,
 max_depth=4,
 min_child_weight=2,
 gamma=0.8,
 subsample=0.65, 
 colsample_bytree=0.45,
 reg_alpha=2,
 objective= 'reg:squarederror',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
xbg3=modelfit(xgb3, x_train1)
xgb1=modelfit(xgb1,x_train1)
from sklearn.ensemble import GradientBoostingRegressor
#Before proceeding further, lets define a function which will help us
#create GBM models and perform cross-validation.
def modelfit1(alg, dtrain, predictors, performCV=True, 
              printFeatureImportance=False, cv_folds=5):
    #Fit the algorithm on the data
    #alg.fit(dtrain[predictors], train['Price'])
    alg.fit(dtrain, predictors)

    #Predict training set:
    #train_predictions = alg.predict(dtrain[predictors])
    dtrain_predictions = alg.predict(dtrain)

    #dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    
    #Perform cross-validation:
    if performCV:
        #cv_score = cross_val_score(alg, dtrain[predictors], dtrain['Price'],
        #cv=cv_folds, scoring='rmse')
        cv_score = cross_val_score(alg, dtrain,predictors, cv=cv_folds, 
                                   scoring='neg_mean_squared_error')
    
    #Print model report:
    print("\nModel Report")
    #print("Accuracy : %.4g" % metrics.accuracy_score(dtrain['is_promoted'].
    #values, dtrain_predictions))
    #print("RMSE (Train): %f" % metrics.mean_squared_error(dtrain['Price'],
    #dtrain_predictions))
    print("RMSE (Train): %f" % metrics.mean_squared_error(predictors, dtrain_predictions))

    
    if performCV:
        print("CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % 
              (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
                
    #Print Feature Importance:
    if printFeatureImportance:
        feat_imp = pd.Series(alg.feature_importances_, predictors).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        return alg
target = "Price"
predictors = [x for x in train.columns if x not in [target]]
gbm0 = GradientBoostingRegressor(random_state=10)
modelfit1(gbm0, x_train, y_train)
param_test1 = {'n_estimators':range(200,600,10)}
gsearch1 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, 
                                                              min_samples_split=500,
                                  min_samples_leaf=50,max_depth=8,max_features='sqrt', 
                                                              subsample=0.8,random_state=10), 
                       param_grid = param_test1, scoring='r2',n_jobs=4,iid=False, cv=5)
gsearch1.fit(x_train,y_train)
param_test2 = {'alpha':[0.5,0.4,0.3]}
gsearch2 = GridSearchCV(estimator = GradientBoostingRegressor(learning_rate=0.1, 
                                                              alpha=0.5,
                                                              n_estimators=590,
                                                              max_leaf_nodes=8,
                                                              max_depth=7,
                                                max_features=8, subsample=0.8, 
                                                              random_state=10,
                                                              min_samples_leaf=21),
                       param_grid = param_test2, scoring='r2',n_jobs=4,iid=False, cv=5)
gsearch2.fit(train[predictors],train[target])
gbm1 = GradientBoostingRegressor(learning_rate=0.1, alpha=0.5,n_estimators=590,
                                 min_samples_split=2, max_leaf_nodes=8,max_depth=7,
                                                max_features=8, subsample=0.8,
                                 random_state=10,min_samples_leaf=21)
gbm1.fit(x_train,y_train)
preds_fin= xgb3.predict(x_test1)

preds_fin1 = xgb1.predict(x_test1)

preds_fin2 = gbm1.predict(x_test)
from sklearn.externals import joblib
from numpy import loadtxt
joblib.dump(xgb3, "xgb3.joblib.dat")
joblib.dump(xgb1, "xgb1.joblib.dat")
joblib.dump(gbm1, "gbm1.joblib.dat")

from sklearn import metrics

# Measure MSE error.  
score = metrics.mean_squared_error(preds_fin,y_test1)
print("Final score (MSE): {}".format(score))

score = metrics.mean_squared_error(preds_fin1,y_test1)
print("Final score (MSE): {}".format(score))

score = metrics.mean_squared_error(preds_fin2,y_test)
print("Final score (MSE): {}".format(score))
import numpy as np
import matplotlib.pyplot as plt

# Measure RMSE error.  RMSE is common for regression.
score = np.sqrt(metrics.mean_squared_error(preds_fin,y_test1))
print("Final score (RMSE): {}".format(score))

score = np.sqrt(metrics.mean_squared_error(preds_fin1,y_test1))
print("Final score (RMSE): {}".format(score))

score = np.sqrt(metrics.mean_squared_error(preds_fin2,y_test))
print("Final score (RMSE): {}".format(score))
def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5
hi = xgb3.predict(x_train1)
hi1 = xgb1.predict(x_train1)
hi2 = gbm1.predict(x_train)
score = rmsle(y_train1.values,hi)
print("Final score (LRMSE): {}".format(1-score))
score = rmsle(y_train1.values,hi1)
print("Final score (LRMSE): {}".format(1-score))
score = rmsle(y_train.values,hi2)
print("Final score (LRMSE): {}".format(1-score))
# Regression chart.
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
from sklearn.externals import joblib
ensemble_pred=[]
for i in range(len(preds_fin1)):
  ensemble_pred.append((preds_fin1[i]+preds_fin2[i]+preds_fin[i])/3)
ensemble_preds = np.array(ensemble_pred,dtype=np.float64)
ensemble_preds

score = rmsle(y_test1.values,preds_fin)
print("Final score (LRMSE): {}".format(1-score))

score = rmsle(y_test1.values,preds_fin1)
print("Final score (LRMSE): {}".format(1-score))
score = rmsle(y_test.values,preds_fin2)
print("Final score (LRMSE):{}".format(1-score))

print("ensemble score")
score = rmsle(y_test.values,ensemble_preds)
print("Final score (LRMSE):{}".format(1-score))