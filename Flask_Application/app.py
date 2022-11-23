import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__) #initialising the flask app
filepath="model_movies.pkl"
model=pickle.load(open(filepath,'rb'))#loading the saved model
scalar=pickle.load(open("scalar_movies.pkl","rb"))#loading the saved scalar file

@app.route('/')
def home():
    return render_template('movieboxoffice.html')

@app.route('/y_predict',methods=['POST'])
def y_predict():
    '''
    For rendering results on HTML
    '''
    input_feature=[float(x) for x in request.form.values()]
    features_values=[np.array(input_feature)]
    feature_name=['budget','genres','popularity','runtime','vote_average','vote_count',
                  'director','release_month','release_DOW']
    x_df=pd.DataFrame(features_values,columns=feature_name)
    x=scalar.transform(x_df)
     # predictions using the loaded model file
    prediction=model.predict(x)
    print("Prediction is:",prediction)
    return render_template("revenuepredict.html",prediction_text=prediction[0])
if __name__ == "__main__":
    app.run(debug=False)
