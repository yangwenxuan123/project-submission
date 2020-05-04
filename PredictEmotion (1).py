from flask import Flask, render_template, request
from pandas import read_csv
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)


def clean_data(data):
    tokenizer = nltk.tokenize.TreebankWordTokenizer()
    stemmer = nltk.stem.SnowballStemmer('english')
    words = nltk.corpus.stopwords.words("english")

    data['cleaned'] = data['final'].apply(
        lambda x: " ".join([stemmer.stem(i) for i in tokenizer.tokenize(x) if i not in words]).lower())
    print(data['cleaned'].head())
    return data


def split_data(data):
    x_train, x_test, y_train, y_test = train_test_split(data['cleaned'], data.label, test_size=0.3)
    return x_train, x_test, y_train, y_test


def load_data():
    data = read_csv('train.txt', sep="\t")
    print(data.head())
    data['final'] = data[['turn1', 'turn2', 'turn3']].apply(lambda x: ' '.join(x), axis=1)
    return data


def get_linear_svm_classifier():
    linear_svm_classifier = Pipeline([('feature_extraction', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                                      ('feature_selection',  SelectKBest(chi2, k=10000)),
                                      ('classifier', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])
    return linear_svm_classifier


def get_svm_classifer():
    svm_classifier = Pipeline([('feature_extraction', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                               ('feature_selection',  SelectKBest(chi2, k=10000)),
                               ('classifier', SVC(kernel='rbf',probability=True))])
    return svm_classifier


def get_decision_tree_classifier():
    decision_tree_classifier = Pipeline([('feature_extraction', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                                         ('feature_selection',  SelectKBest(chi2, k=10000)),
                                         ('classifier', DecisionTreeClassifier(random_state=0))])
    return decision_tree_classifier


def get_logistic_regression_classifier():
    logistic_regression_classifier = Pipeline([('feature_extraction', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                                               ('feature_selection',  SelectKBest(chi2, k=10000)),
                                               ('classifier', LogisticRegression(random_state=0,max_iter=50,penalty='l2'))])
    return logistic_regression_classifier


def ensemble_classifiers():
    linear_svm_classifier = get_linear_svm_classifier()
    svm_classifier = get_svm_classifer()
    decision_tree_classifier = get_decision_tree_classifier()
    logistic_regression_classifier = get_logistic_regression_classifier()

    ensemble_classifier = Pipeline([['ensemble_classifier',
                                     VotingClassifier(estimators=[("linear_svm_classifier", linear_svm_classifier),
                                                                  ("svm_classifier", svm_classifier),
                                                                  ("decision_tree_classifier", decision_tree_classifier),
                                                                  ("logistic_regression_classifier", logistic_regression_classifier)])]])
    return ensemble_classifier


data = load_data()
cleaned_data = clean_data(data)

x_train, x_test, y_train, y_test = split_data(cleaned_data)
model = ensemble_classifiers().fit(x_train, y_train)

"""
Accuracy test block

linear_svm_classifier = get_linear_svm_classifier()
svm_classifier = get_svm_classifer()
decision_tree_classifier = get_decision_tree_classifier()
logistic_regression_classifier = get_logistic_regression_classifier()

for clf, label in zip([logistic_regression_classifier, decision_tree_classifier, svm_classifier , linear_svm_classifier, ensemble_classifiers()], ['Logistic Regression', 'DecisionTree', 'Support Vector Mchine' , 'Linear SVM', 'Ensemble']):
    scores = cross_val_score(clf, x_train, y_train, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean()*100, scores.std()*100, label))"""


def predict_emotion(msgs):
    emotion_label = model.predict([msgs])
    print(emotion_label)
    return emotion_label


@app.route("/",methods=["GET","POST"])
def home():
    return render_template("GUI.html")


@app.route("/predict",methods=["GET","POST"])
def predict():
    form = request.get_json(force=True)
    #print("Request form is: "+request.form["0"])
    msgs= []
    if request.method == 'POST':  # this block is only entered when the form is submitted
        for i in form:
            print(i)
    
    msgs = " ".join(form)
    print("final msg :" + msgs)
    prediction = predict_emotion(msgs)
    print(prediction)
    return prediction[0]


if __name__ == "__main__":
    app.run()
