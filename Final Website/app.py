from flask import Flask, flash, request, url_for, jsonify, render_template, redirect, send_from_directory
from werkzeug.utils import secure_filename

import pandas as pd
import datetime as dt
import json
import numpy as np
import os

# Upload Locations
UPLOAD_FOLDER = './static/media'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'mov', 'img'}

# Flask Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route("/")
# def home():
#     return render_template("upload_file.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
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
    return render_template("upload_file.html")
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


# @app.route("/black&white")
# def games():
#     data_list = []
#     for row in session.query(top_games).all():
#         data_list.append(row.__dict__["game_name"])
#     # return data_list
#     return render_template("icons.html", list=data_list)


# @app.route("/linear_optical")
# def stats():
#     return render_template("dashboard.html")

# @app.route("/density_optical")
# def vid_stats():
#     return render_template("video.html")


if __name__ == "__main__":
    app.run(debug=True)