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
import json

df = pd.read_csv("./fake_real_master_dataset.csv")
fk_arr = []


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text


for i in range(df.shape[0]):
    if df['fake'][i] == 0:
        fk_txt = df['text'][i]
        fk_txt = wordopt(fk_txt)
        fk_txt = fk_txt.replace('\r', '')
        fk_txt = fk_txt.replace(' ', '')
        fk_arr.append(fk_txt)

fk_arr = np.array(fk_arr)


def ValuePredictor(news_text):
    if news_text in fk_arr:
        return 0
    else:
        return 1


app = Flask(__name__)


@app.route('/', methods=['POST'])
def result():
    to_predict_list = request.form.to_dict()
    user_news = to_predict_list['news_text']
    user_news = wordopt(user_news)
    user_news = user_news.replace(' ', '')

    result = ValuePredictor(user_news)
    if(result == 0):
        ret = {'output': 'FAKE News'}
        ret = json.dumps(ret)
        print(ret)
        return ret
    else:
        ret = {'output': 'NOT FAKE News'}
        ret = json.dumps(ret)
        return ret


if __name__ == '__main__':
    app.run()

# Heroku endpoint:
# https://flask-fake-news-api.herokuapp.com/
