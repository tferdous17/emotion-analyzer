from flask import Flask, render_template
from views import views

app = Flask(__name__)
app.register_blueprint(views, url_prefix="/analyzer")

@app.route("/")
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)