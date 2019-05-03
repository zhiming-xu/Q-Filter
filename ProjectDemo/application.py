from flask import Flask, render_template, url_for, request  # import main Flask class and request object

application = app = Flask(__name__, template_folder='template')  # create the Flask app

posts = {
        'author': 'Let\'s try our model with your questions!',
        'title': 'Type in Your Question',
        'question': 'who is the president of the united states?',
        'opinion': 'Good',
        'resultBERT': '',
        'resultCNN': ''
}

try:
    from interface import predict_bert
except IOError:
    posts['author'] = 'An error occured trying to read the file.'
   
except ImportError:
    posts['author'] = "NO module found"
  
except:
    posts['author'] = 'error occur when loading model'

@app.route("/")
@app.route("/home")
def home():
    return render_template('home.html', posts=posts)


@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['question']
    processed_text = text.capitalize()
    if processed_text:
        posts['question'] = processed_text
    else:
        # we use the default question
        processed_text = posts['question']
    posts['resultBERT'] = "Bad question" \
                          if predict_bert([processed_text]) == 1 \
                          else "Valuable question"
    posts['opinion'] = request.form['opinion'].capitalize()
    return render_template('home2.html', posts=posts)


@app.route("/about")
def about():
    return render_template('about.html', title='About')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
