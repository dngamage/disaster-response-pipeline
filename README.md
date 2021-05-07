# Data Science Nanodegree

# disaster-response-pipeline

## Description
This Project is part of Data Science Nanodegree Program conducted by Udacity in collaboration with **Figure Eight**. The dataset contains pre-labelled tweets and messages from real-life disaster events. The project aim is to build a Natural Language Processing (NLP) model to categorize messages on a real time basis.

The best performing machine learning model will be deployed as a web app where the user can test their own tentative messages to see how they would be classified with the models I selected and trained. Through the web app the user can also consult visualizations of the clean and transformed data.

This project is divided in the following key sections:

- Processing data, building an ETL pipeline to extract data from source, clean the data and save them in a SQLite DB
- Build a machine learning pipeline to train the which can classify text message in various categories
- Run a web app which can show model results in real time

# Getting Started

## Dependencies
- Python 3.5+
- Natural Language Process Libraries: NLTK
- SQLlite Database Libraries: SQLalchemy
- Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
- Model Loading and Saving Library: Pickle
- Web App and Data Visualization: Flask, Plotly

# Project Git URL
https://github.com/dngamage/disaster-response-pipeline

# Executing Program
1. can run the following commands in the project's directory to set up the database, train model and save the model.

    - To run ETL pipeline to clean data and store the processed data in the database python 
    
           process_data.py  data/disaster_messages.csv  data/disaster_categories.csv  data/disaster_response_db.db
    - To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file python 
    
           train_classifier.py  data/disaster_response_db.db  models/classifier.pkl
2. Run the following command in the app's directory to run your web app. 

          python run.py

3. Go to http://0.0.0.0:3001/
   If you are running local machin then use http://127.0.0.1:3001/
   
# More valueable  Files
app/templates/*: templates/html files for web app

- data/process_data.py: Extract Train Load (ETL) pipeline used for data cleaning, feature extraction, and storing data in a SQLite database

- models/train_classifier.py: A machine learning pipeline that loads data, trains a model, and saves the trained model as a .pkl file for later use

- run.py: This file can be used to launch the Flask web app used to classify disaster messages

# Acknowledgements
- [Udacity](https://www.udacity.com//) for providing an amazing Data Science Nanodegree Program with proper training
- [Figure Eight](https://appen.com/) for providing the relevant dataset to train the model

# Screenshots 

![Home](/screenshots/01.png?raw=true "")
![Home](/screenshots/02.png?raw=true "")
![Home](/screenshots/03.png?raw=true "")
![Home](/screenshots/07.png?raw=true "")
![Home](/screenshots/08.png?raw=true "")
![Home](/screenshots/04.png?raw=true "")
![Home](/screenshots/06.png?raw=true "")
![Home](/screenshots/05.png?raw=true "")

### From Udacity Server
![Home](/screenshots/09.png?raw=true "")
![Home](/screenshots/10.png?raw=true "")

