# from Fake_news import script

# app = create_app()

# if __name__ == '__main__':
#     app.run()

import numpy as np
import pickle
import re
import string
import pandas as pd
from flask import Flask, request, render_template 

loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
loaded_vectorizer = pickle.load(open('pikle_vectorizer.pkl', 'rb'))

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = loaded_vectorizer.transform(new_x_test)
    pred_GBC = loaded_model.predict(new_xv_test)
    return pred_GBC[0]
# prediction function
def ValuePredictor(news_text):
	result = manual_testing(news_text)
	return result

app = Flask(__name__)
@app.route('/', methods = ['POST'])
def result():
    to_predict_list = request.form.to_dict()
    result = ValuePredictor(to_predict_list['news_text'])	
    if(result==0):
        return "FAKE News"
    else:
        return "NOT FAKE News"

if __name__ == '__main__':
    app.run()