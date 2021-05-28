from __future__ import print_function
from __future__ import division
from flask import Flask, flash, request, url_for, jsonify, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os
import image_view

# Upload Locations
UPLOAD_FOLDER = './static/media'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mov', 'img'}

# Flask Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template("upload.html")


    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    image_view.black_white(filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               'filename2.jpg')



if __name__ == "__main__":
    app.run(debug=True)