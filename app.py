from flask import Flask, render_template, request
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import os
key = os.environ.get('AZURE_TEXT_ANALYTICS_KEY')
endpoint = os.environ.get('AZURE_TEXT_ANALYTICS_ENDPOINT')


app = Flask(__name__)
# Define a custom filter to find the maximum value from a list of values
def max_value(lst):
    if isinstance(lst, list):
        return max(lst)
    else:
        return lst

# Register the custom filter in the Jinja environment
app.jinja_env.filters['max_value'] = max_value

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_text():
    text = request.form['text']

    # Initialize Azure Text Analytics client
    key = 'f653569c1a074b5f9bd1e76e63c8c999'
    endpoint = 'https://bitthal.cognitiveservices.azure.com/'
    credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint, credential)

    try:
        # Perform sentiment analysis with opinion mining
        result = text_analytics_client.analyze_sentiment(documents=[text], show_opinion_mining=True)
        
        # Process sentiment analysis result for each sentence
        analyzed_sentences = []
        for sentiment_response in result:
            for sentence in sentiment_response.sentences:
                sentiment = sentence.sentiment
                confidence = sentence.confidence_scores[sentiment]

                # Process opinion mining result for each sentence
                opinion_data = []
                if sentence.mined_opinions:
                    for opinion in sentence.mined_opinions:
                        target = opinion.target
                        opinion_sentiment = sentence.sentiment
                        sentiment_confidence = sentence.confidence_scores[opinion_sentiment]
                        opinion_data.append({'target': target, 'sentiment': opinion_sentiment, 'confidence': sentiment_confidence})

                analyzed_sentences.append({'sentence_text': sentence.text, 'sentiment': sentiment, 'confidence': confidence, 'opinion_data': opinion_data})

        return render_template('result.html', analyzed_sentences=analyzed_sentences)
    except Exception as e:
        print("Exception:", e) # Print exception for debugging
        return render_template('error.html', message=str(e))

if __name__ == '__main__':
    app.run(debug=True)
