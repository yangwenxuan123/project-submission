from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse  
from allennlp.predictors.predictor import Predictor as AllenNLPPredictor

predictor = AllenNLPPredictor.from_path("https://storage.googleapis.com/allennlp-public-models/bidaf-elmo-model-2018.11.30-charpad.tar.gz")

app = Flask(__name__)




@app.route('/bot', methods=['POST'])
def bot():
    incoming_msg = request.values.get('Body', '').lower()
    resp = MessagingResponse()
    msg = resp.message()
    responded = False
    predictor
    
    passage = "Dravid was born in a Marathi family in Indore, Madhya Pradesh. His family later moved to Bangalore, Karnataka, where he was raised. His mother tongue is Marathi. Dravid's father Sharad Dravid worked for a company that makes jams and preserves, giving rise to the later nickname Jammy. His mother, Pushpa, was a professor of architecture at the University Visvesvaraya College of Engineering (UVCE), Bangalore. Dravid has a younger brother named Vijay. He did his schooling at St. Joseph's Boys High School, Bangalore and earned a degree in commerce from St. Joseph's College of Commerce, Bangalore. He was selected to India national cricket team while studying MBA in St Joseph's College of Business Administration. He is fluent in several languages, Marathi, Kannada, English and Hindi"
    
    result = predictor.predict(passage = passage, question = incoming_msg )
    msg.body(result["best_span_str"])
    
    return str(resp)