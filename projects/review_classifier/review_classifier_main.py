from flask import Flask, escape, request
from flask import jsonify
from flask import render_template
from fastai.text import *
from pathlib import Path
from flask import send_from_directory
import fastai

defaults.device = torch.device('cpu')

app = Flask(__name__, static_folder='text-classifier-ui/static', static_url_path = '/static', template_folder='text-classifier-ui')

bs=12
path = Path('/root/.fastai/data/yelp_review_full_csv')
data_clas = load_data(path, 'data_clas.pkl', bs=bs)
data_lm = load_data(path, 'data_lm.pkl', bs=bs)
learn = text_classifier_learner(data_clas, AWD_LSTM, drop_mult=0.5)
learn_lm = language_model_learner(data_lm, AWD_LSTM, drop_mult=0.3)
learn.load_encoder('fine_tuned_enc')
learn.load('fully_trained')
learn.model.training = False
learn.model = learn.model.cpu()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    sentence = request.args.get("sentence", "")
    pred_class, pred_idx, outputs = learn.predict(sentence)
    response = jsonify({"response": sentence, "prediction": str(learn.predict(sentence)), "pred_class": int(str(pred_class))})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/generate')
def generate():
    text = request.args.get("text", "")
    n_words = request.args.get("n_words", "120")
    n_sentences = request.args.get("n_sentences", 1)
    result = (learn_lm.predict(text, int(n_words), temperature=0.75) for _ in range(n_sentences))
    response = jsonify({"text": "\n".join(result)})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/assets/images/<path:filename>')
def custom_static(filename):
    return send_from_directory('text-classifier-ui/assets/images', filename)

