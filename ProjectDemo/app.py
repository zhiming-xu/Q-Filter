from flask import Flask, render_template, url_for, request  # import main Flask class and request object
import test

app = Flask(__name__, template_folder='template')  # create the Flask app

posts = {
        'author': 'Let\'s try our model with your questions!',
        'title': 'Type in Your Question',
        'question': '',
        'opinion': '',
        'resultBERT': '',
        'resultCNN': ''
}


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['question']
    processed_text = text.capitalize()
    posts['question'] = processed_text
    posts['resultBERT'] = test.testBERT(processed_text)
    posts['resultCNN'] = test.testCNN(processed_text)
    posts['opinion'] = request.form['opinion'].capitalize()
    return render_template('home2.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # run app in debug mode on port 5000
