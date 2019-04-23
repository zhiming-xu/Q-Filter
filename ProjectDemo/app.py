from flask import Flask, render_template, url_for, request  # import main Flask class and request object
import test

app = Flask(__name__, template_folder='template')  # create the Flask app

posts = {
        'author': 'Let\'s try our model with your questions!',
        'title': 'Type in Your Question',
        'question': '',
        'result': ''
}


@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = text.capitalize()
    posts['question'] = processed_text
    posts['result'] = test.test(processed_text)
    return render_template('home2.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


if __name__ == '__main__':
    app.run(debug=True, port=5000)  # run app in debug mode on port 5000
