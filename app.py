import chatbot
from flask import Flask, render_template, request
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/get')
def chatbot_response():
    message = request.args.get('msg')
    ints = chatbot.predict_class(message)
    res = chatbot.get_response(ints, chatbot.intents)
    return res


if __name__ == "__main__":
    app.run()