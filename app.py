from flask import Flask, render_template, request
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Download NLTK resources if missing
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open('model_sms_spam.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer_sms.pkl', 'rb'))

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):

    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []

    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    message = request.form['message']

    transformed_sms = transform_text(message)

    vector_input = vectorizer.transform([transformed_sms])

    prediction = model.predict(vector_input)[0]

    confidence = round(
        max(model.predict_proba(vector_input)[0]) * 100,
        2
    )

    if prediction == 1:
        result = "🚨 SPAM DETECTED"
        result_class = "spam"
    else:
        result = "✅ SAFE MESSAGE"
        result_class = "ham"

    return render_template(
        'index.html',
        prediction=result,
        result_class=result_class,
        confidence=confidence,
        message=message
    )

if __name__ == '__main__':
    app.run(debug=True)