from flask import Flask, request, render_template
import nltk
import re
import csv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.classify import NaiveBayesClassifier

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    
    #convert to lowercase
    text1 = request.form['text1'].lower()
    
    # text_final = ''.join(c for c in text1 if not c.isdigit())
        
    # #remove stopwords    
    # processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    # sa = SentimentIntensityAnalyzer()
    # dd = sa.polarity_scores(text=processed_doc1)
    # compound = round((1 + dd['compound'])/2, 2)
    
    
    #######################################################################
    
    # load email data from CSV file
    data_path = 'emails.csv'
    messages = []
    with open(data_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            messages.append(row[0])
            
    # function to clean text by removing hyperlinks and stopwords
    def clean_text(text):
        # text = re.sub(r'http\S+', '', text)  # remove hyperlinks
        # text = re.sub(r'\b\w{1,3}\b', '', text)  # remove short words
        # text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
        text = text.lower()  # convert to lowercase
        tokens = word_tokenize(text)  # tokenize text
        stop_words = set(stopwords.words('english'))  # get stop words
        filtered_tokens = [token for token in tokens if token not in stop_words]  # remove stop words
        cleaned_text = ' '.join(filtered_tokens)  # join tokens back into string
        return cleaned_text
    
    
    analyzer = SentimentIntensityAnalyzer()
    
    
    # clean email messages and extract features for Naive Bayes classifier
    featuresets = []
    for message in messages:
        cleaned_message = clean_text(message)
        scores = analyzer.polarity_scores(cleaned_message)
        features = {
            'positive_score': scores['pos'],
            'negative_score': scores['neg'],
            'neutral_score': scores['neu'],
            'compound_score': scores['compound']
        }
        
    
        featuresets.append((features, 'positive' if scores['compound'] >= 0.05 else 'negative' if scores['compound'] <= -0.05 else 'neutral'))
        
        # train Naive Bayes classifier on featuresets
    classifier = NaiveBayesClassifier.train(featuresets)

    # test classifier on new email message
    # new_message = "Hi John, just wanted to touch base with you about the project we discussed last week. i also have surprised for you too"
    cleaned_new_message = clean_text(text1)
    scores = analyzer.polarity_scores(cleaned_new_message)
    features = {
        'positive_score': scores['pos'],
        'negative_score': scores['neg'],
        'neutral_score': scores['neu'],
        'compound_score': scores['compound']
    }
    classification = classifier.classify(features)
    
    # print("Positive score:", scores['pos'])
    # print("Negative score:", scores['neg'])
    # print("Neutral score:", scores['neu'])
    # print("Compound score:", scores['compound'])
    # print("Naive Bayes classification:", classification)
        
    

    return render_template('form.html', final=scores['compound'], text1=text1,text2=scores['pos'],text5= scores['neg'],text4=scores['compound'],text3=scores['neu'])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5002, threaded=True)
