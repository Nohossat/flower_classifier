from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
import os
from fastai.vision import *

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# import model
path = Path('')
learn = load_learner(path)
defaults.device = torch.device('cpu')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index(response = None, filename = None):
    filename = request.args.get('filename')
    response = request.args.get('response')
    return render_template('index.html', response = response, filename = filename)

@app.route('/predict', methods=['POST'])
def predict():
    # return prediction
    if 'img' not in request.files:
        response = 'No file part'
        return redirect(url_for('/', response=response))

    img = request.files['img']
    if img.filename == '':
        response = 'No selected file'
        return redirect(url_for('/', response=response))

    if img and allowed_file(img.filename):
        filename = secure_filename(img.filename)
        img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        url_file = url_for('static', filename=f'uploads/{filename}')
        img_obj = open_image(path/'static'/'uploads'/filename)

        # predict
        pred_class, pred_idx, outputs = learn.predict(img_obj)
        return redirect(url_for('index', response=pred_class.obj, filename=url_file))


if __name__ == "__main__":
    app.run(port=5000, host="0.0.0.0", debug=True)

