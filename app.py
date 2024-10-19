import os
import tensorflow as tf
import numpy as np
from tf_keras.preprocessing import image
from PIL import Image
import cv2
from tf_keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model = load_model("BrainTumor10Epochs_categorical.h5")
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(class_num):
	if class_num==0:
		return "YAYY!! No Brain Tumor Found!!"
	elif class_num==1:
		return ":( We found a Tumor :("


def getResult(pic):
    image=cv2.imread(pic)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
    final_result = np.argmax(result, axis=1)
    return final_result


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        result=get_className(value) 
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)