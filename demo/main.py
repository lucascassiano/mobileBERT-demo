# https://github.com/gemde001/MobileBERT

from flask import Flask, request, render_template
from mobilebert import MobileBERT
import wikipedia

bert = MobileBERT()
# answer = bert.run(
#     question,
#     content
# )

# f = open('content.txt', 'r')
# content = f.read()
# f.close()

app = Flask(__name__)


@app.route('/api', methods=['GET'])
def run():
    question = request.args.get('question', default=None, type=str)
    content = request.args.get('content', default=None, type=str)
    topic = request.args.get('topic', default=None, type=str)

    print(question)
    # print(content)
    # c_search = wikipedia.page(search)
    # print(wikipedia.summary(search))

    answer = bert.run(
        question,
        content
    )

    print(answer)

    return {"answer": answer}


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")


app.run(debug=True, host="0.0.0.0")
