from flask import Flask, jsonify, render_template,redirect
import pandas as pd
import datetime as dt
import json
import numpy as np

# Python SQL toolkit and Object Relational Mapper
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func

# Create engine to hawaii.sqlite
engine = create_engine("sqlite:///./static/sqlite/twitch.sqlite")

# Reflect an existing database into a new model
Base = automap_base()

# Reflect the tables
Base.prepare(engine, reflect=True)

# Save references to each table
game_data = Base.classes.game_data
top_games = Base.classes.top_games

# Create our session (link) from Python to the DB
session = Session(engine)

# Flask Setup
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("user.html")
    
    # return (
    #     "<h1>Welcome to the Twitch Dashboard Homepage!</h1><br/>"
    #     "Available Routes:<br/>"
    #     "/top_games<br/>"
    #     "/platforms<br/>"
    #     "/timeline<br/>"
    #     "/hours<br/>"
    #     "/growth<br/>"
    #     "/growth/games<br/>"
    #     "/growth/streamers<br/>"
        
    # )


@app.route("/top_games")
def games():
    data_list = []
    for row in session.query(top_games).all():
        data_list.append(row.__dict__["game_name"])
    # return data_list
    return render_template("icons.html", list=data_list)


@app.route("/game_stats")
def stats():
    return render_template("dashboard.html")

# @app.route("/tables")
# def tables():
#     return render_template("tables.html")

@app.route("/video_stats")
def vid_stats():
    return render_template("video.html")

# @app.route("/bubbles")
# def bubbles():
#     return render_template("bubbles.html")

# @app.route("/timeline")
# def tobs():
#     # Query the dates and temperature observations of the most active station for the last year of data
#     prev_year = dt.date(2017, 8, 23) - dt.timedelta(days=365)
#     results = session.query(Measurement.date, Measurement.tobs).filter(Measurement.station == 'USC00519281')\
#                 .filter(Measurement.date >= prev_year).all()
        
#     session.close()    
#     temps = {date: tobs for date, tobs in results}

#     # Return a JSON list of temperature observations (TOBS) for the previous year.
#     date_temp = list(np.ravel(temps))
#     return jsonify(date_temp)

# @app.route("/hours")
# def tobs():
#     # Query the dates and temperature observations of the most active station for the last year of data
#     prev_year = dt.date(2017, 8, 23) - dt.timedelta(days=365)
#     results = session.query(Measurement.date, Measurement.tobs).filter(Measurement.station == 'USC00519281')\
#                 .filter(Measurement.date >= prev_year).all()
        
#     session.close()    
#     temps = {date: tobs for date, tobs in results}

#     # Return a JSON list of temperature observations (TOBS) for the previous year.
#     date_temp = list(np.ravel(temps))
#     return jsonify(date_temp)

# @app.route("/growth")
# def tobs():
#     # Query the dates and temperature observations of the most active station for the last year of data
#     prev_year = dt.date(2017, 8, 23) - dt.timedelta(days=365)
#     results = session.query(Measurement.date, Measurement.tobs).filter(Measurement.station == 'USC00519281')\
#                 .filter(Measurement.date >= prev_year).all()
        
#     session.close()    
#     temps = {date: tobs for date, tobs in results}

#     # Return a JSON list of temperature observations (TOBS) for the previous year.
#     date_temp = list(np.ravel(temps))
#     return jsonify(date_temp)

# @app.route("/growth/games")
# def tobs():
#     # Query the dates and temperature observations of the most active station for the last year of data
#     prev_year = dt.date(2017, 8, 23) - dt.timedelta(days=365)
#     results = session.query(Measurement.date, Measurement.tobs).filter(Measurement.station == 'USC00519281')\
#                 .filter(Measurement.date >= prev_year).all()
        
#     session.close()    
#     temps = {date: tobs for date, tobs in results}

#     # Return a JSON list of temperature observations (TOBS) for the previous year.
#     date_temp = list(np.ravel(temps))
#     return jsonify(date_temp)

# @app.route("/growth/streamers")
# def tobs():
#     # Query the dates and temperature observations of the most active station for the last year of data
#     prev_year = dt.date(2017, 8, 23) - dt.timedelta(days=365)
#     results = session.query(Measurement.date, Measurement.tobs).filter(Measurement.station == 'USC00519281')\
#                 .filter(Measurement.date >= prev_year).all()
        
#     session.close()    
#     temps = {date: tobs for date, tobs in results}

#     # Return a JSON list of temperature observations (TOBS) for the previous year.
#     date_temp = list(np.ravel(temps))
#     return jsonify(date_temp)



if __name__ == "__main__":
    app.run(debug=True)