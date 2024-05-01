from flask import Flask, render_template, request
from model import translator

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate_text():
    text = request.form['text']
    source = 'flutter'
    target = 'react'
    # TODO: Translate Text
    translated_text = ""
    return render_template('result.html', translated_text=translated_text)

if __name__ == '__main__':
    app.run(debug=True)
