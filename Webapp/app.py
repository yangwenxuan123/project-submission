from flask import Flask, request, jsonify, render_template
import joblib
import sklearn as sk
import pickle
import torch
import torchtext
#import nltk
#nltk.download('punkt')
from nmtmodel import NMTModel
from nltk.tokenize import word_tokenize
from torchtext import vocab
try:
    vocab._default_unk_index
except AttributeError:
    def _default_unk_index():
        return 0
    vocab._default_unk_index = _default_unk_index

app = Flask(__name__)
model = joblib.load('model.pkl')

vocab = pickle.load(open('/Users/sk/Desktop/flaskui/vocab', 'rb'))
modelnmt = NMTModel(len(vocab), embedding_dim=100, hidden_dim=256, vocab=vocab, device = torch.device('cpu'), bidirectional=True)
modelnmt.load_state_dict(torch.load('/Users/sk/Desktop/flaskui/nmtmodel2.pt', map_location= torch.device('cpu')))
modelnmt.eval()


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    tmp = [str(x) for x in request.form.values()]
    input_txt = tmp[0]
    #extract features
    features = model['vect'].get_feature_names()
    #Sorting based on tf_idf values in the vector and taking top 10 features
    def sort_matrix(matrix):
        col_data = zip(matrix.col, matrix.data)
        return sorted(col_data, key= lambda x: (x[1], x[0]), reverse=True)
    
    def take_topn(features, col_data, n):
        top_col_data = col_data[:n]
        result = {}
        for col, tfidf in top_col_data:
            feature = features[col]
            result[feature] = tfidf
        return result
    #checking sort and top features function on a sample
    sample_vector = model.transform([input_txt])
    vec = sort_matrix(sample_vector.tocoo())
    output_tf = list(take_topn(features, vec, 5).keys())
    output_tf1 = output_tf[0]+', '+output_tf[1] +', '+output_tf[2]

    # nn model
    inp = word_tokenize(tmp[0])
    inp = [vocab.stoi[word] if word in vocab.stoi else vocab.stoi['<unk>'] for word in inp]
    enc_output = modelnmt.encoder(torch.tensor(inp).unsqueeze(0))

    res = modelnmt.decoder.infer_rnn_auto_regressive(encoder_output_dict=enc_output,vocab=vocab, length= 3).view(-1).detach().cpu().numpy()

    output_nn = ''
    for i in res:
        output_nn = output_nn+', '+vocab.itos[int(i)]
    
    return render_template('index.html', prediction_text_tf='Top 3 key phrases:  {}'.format(output_tf1), prediction_text_nn='Top 3 key phrases:  {}'.format(output_nn[1:]))

