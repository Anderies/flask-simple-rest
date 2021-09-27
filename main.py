from flask import Flask, jsonify, request  # import objects from the Flask model
from keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
# import torch
app = Flask(__name__)  # define app using Flask

tweets = [{'content': 'dont know what todo', 'date': '17-02-2021', 'id': '1'},
          {'content': 'doing flask and its pretty great', 'date': '18-02-2021', 'id': '2'}]

# print(keras.__version__)
# model = load_model('./tf_model.h5')

# NO PYTORCH
tokenizer = AutoTokenizer.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english")

pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer)


@app.route('/', methods=['GET'])
def test():
    content = request.json['content']

    hasil = pipe(content)
    print(hasil[0]['label'])
    return jsonify({'message': content, 'sentiment': hasil[0]['label']})


@app.route('/tweet', methods=['GET'])
def returnAll():
    return jsonify({'tweets': tweets})


@app.route('/tweet/<string:date>', methods=['GET'])
def returnOne(date):
    tweet = [t for t in tweets if t['date'] == date]
    return jsonify({'data': tweet[0]})


@app.route('/tweet', methods=['POST'])
def createTweet():
    content = request.json['content']
    hasil = pipe(content)

    tweet = {'content': content,
             'date': request.json['date'], 'id': request.json['id'], 'sentiment': hasil[0]['label']}

    tweets.append(tweet)
    return jsonify({'data': tweet})


@app.route('/tweet/<string:id>', methods=['PUT'])
def editOne(id):
    tweet = [t for t in tweets if t['id'] == id]
    tweet[0]['content'] = request.json['content']
    return jsonify({'data': tweet[0]})


@app.route('/delete-tweet/<string:id>', methods=['DELETE'])
def removeOne(id):

    tweet = [t for t in tweets if t['id'] == id]
    tweets.remove(tweet[0])
    return jsonify({'tweets': tweets})


if __name__ == '__main__':
    app.run(debug=True, port=8080)  # run app on port 8080 in debug mode
