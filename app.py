import base64
import json
import urllib.request as urllib
from io import BytesIO
from urllib.parse import quote

import matplotlib.pyplot as plt
import numpy as np
from bert_serving.client import BertClient
from flask import Flask, request
from flask_cors import CORS
from matplotlib.font_manager import FontProperties
from sklearn.manifold import TSNE

myfont = FontProperties(fname=r'C:\\Windows\\Fonts\\msjhbd.ttf')
app = Flask(__name__)
CORS(app)

bc = BertClient()

def http_request(url):
    with urllib.urlopen(url) as content:
        data = content.read()
        return json.loads(data.decode('utf-8'))

def get_synonyms_by_keyword(keyword):
    url = "http://127.0.0.1:8079/api/synonym?keyword={}".format(quote(keyword))
    return http_request(url)

def get_tw_news(keyword, is_tw_2016, is_tw_2017, is_tw_2018):
    url = "http://127.0.0.1:8080/api/news?keyword={}&is_tw_2016={}&is_tw_2017={}&is_tw_2018={}".format(quote(keyword), is_tw_2016, is_tw_2017, is_tw_2018)
    return http_request(url)

def get_hk_news(keyword):
    url = "http://127.0.0.1:8078/api/news?keyword={}".format(quote(keyword))
    return http_request(url)

def get_cn_news(keyword):
    url = "http://127.0.0.1:8077/api/news?keyword={}".format(quote(keyword))
    return http_request(url)

def get_news_sentences(keyword, is_tw, is_hk, is_cn, is_tw_2016, is_tw_2017, is_tw_2018):
    news_list = []
    if is_tw == 'true':
        news_list.extend(set(get_tw_news(keyword, is_tw_2016, is_tw_2017, is_tw_2018)))
    if is_hk == 'true':
        news_list.extend(set(get_hk_news(keyword)))
    if is_cn == 'true':
        news_list.extend(set(get_cn_news(keyword)))
    return list(set(news_list))


def remove_empty_sentence(sentences):
    return list(filter(lambda s: len(s.strip()), sentences))

def seg_sentence_by_punc(content):
    res = []
    n = len(content)
    i = 0
    j = 0
    puncs = ["，", "！", "。", "；", "？"]

    while j < n:
        item = content[j]
        if item.strip() in puncs:
            res.append(content[i:j])
            i = j + 1
        j += 1

    res.append(content[i:j])
    return res

def is_arr_has_term(arr, term):
    if term in arr:
        return True
    else:
        return False

def filter_list_without_term(arr_list, term):
    return list(filter(lambda arr: is_arr_has_term(arr, term), arr_list))

def get_all_sentences(news_list, keyword):
    res = set()
    for news in news_list:
        sentences = seg_sentence_by_punc(news)
        res.update(filter_list_without_term(sentences, keyword))
    return list(res)

def get_single_vector(keyword):
    return bc.encode([keyword])[0]

def fetch_vector_from_news(news_list, keyword):
    if not news_list:
        return get_single_vector(keyword)
    else:
        sentences = get_all_sentences(news_list, keyword)
        if not sentences:
            return get_single_vector(keyword)
        vectors = bc.encode(sentences)
        return np.average(vectors, axis=0)

def visualise_vectors(terms, vectors):
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(terms[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     fontproperties=myfont,
                     ha='right',
                     va='bottom')

    save_file = BytesIO()
    plt.savefig(save_file, format='png')
    save_file_base64 = base64.b64encode(save_file.getvalue()).decode('utf8')
    return save_file_base64

@app.route('/api/base64')
def get_vector_by_keyword():
    keyword = request.args.get("keyword")
    is_tw = request.args.get("is_tw", "true")
    is_hk = request.args.get("is_hk", "true")
    is_cn = request.args.get("is_cn", "true")
    is_tw_2016 = request.args.get("is_tw_2016", "true")
    is_tw_2017 = request.args.get("is_tw_2017", "true")
    is_tw_2018 = request.args.get("is_tw_2018", "true")
    synonyms = get_synonyms_by_keyword(keyword)

    vectors = []
    for synonym in synonyms:
        news_list = get_news_sentences(synonym, is_tw, is_hk, is_cn, is_tw_2016, is_tw_2017, is_tw_2018)
        vector = fetch_vector_from_news(news_list, synonym)
        vectors.append(vector)
    return visualise_vectors(synonyms, vectors)


if __name__ == '__main__':
    app.run()
