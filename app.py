# -*- coding: utf-8 -*-

from flask import Flask, render_template, request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'spam-sms-model.pkl'
spam_model = pickle.load(open(filename, 'rb'))
Xv= pickle.load(open('X-transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
    	message = request.form['message']
    	data = [message]
    	vect = Xv.transform(data).toarray()
    	my_prediction = spam_model.predict(vect)
    	return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
	app.run(debug=True)