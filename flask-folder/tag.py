from flask import Flask, render_template
from flask import request, redirect
from some import main

app = Flask(__name__)

@app.route('/')
def tag():
    author = "Me"
    name = "You"
    return render_template('./index.html', author=author, name=name)

@app.route('/createpost', methods = ['POST'])
def requesttag():
    text = request.form['articlebody']
    tags = main(text)
    return render_template('/displaypost.html',text=text,tags=tags)

@app.route('/displaypost.html')
def display():
    return render_template('displaypost.html',text=text,tags=tags)

if __name__ == "__main__":
    app.run()
