import joblib, numpy as np
scaler=joblib.load('models/scaler_reg.joblib')
model=joblib.load('models/model_regression.joblib')
features=np.array([[50000,10.0,2,0,1,5]])
fs=scaler.transform(features)
pred=model.predict(fs)
print('raw prediction:',pred[0])
print('as kFCFA ->', int(round(pred[0]*1000)))
val = int(round(pred[0]*1000))
formatted = "{:,}".format(val).replace(",", " ") + ' FCFA'
print('formatted ->', formatted)
