# Disaster response pipeline

This project is developed as part of the udacity nanodegree program.
The task is to build an application that can classify a given disaster message into various categories
Example: Screenshot of final application
![prediction page](plots/3.png) 

Project summary:
* Using a dataset of real messages that were sent during disaster events, a machine learning pipeline is developed to categorize these events so that the message can be forwarded to an appropriate disaster relief agency.
* A web application is also developed where an emergency worker can input a new message and get classification results in several categories.
* Visualizations of the data is also present.
* Datasets provided by Figure 8, Udacity

Some of the graphs are: 
1. 
![prediction page](plots/1.png) 

2.
![prediction page](plots/2.png) 


Files:

* ELT Pipeline Preparation notebook - Documents the data preperation process from two CSV files: messages.csv and categories.csv (in data folder)
* ML Pipeline Preparation notebook - Documents the steps and various techniques tried while creating the machine learning pipeline to get a model that can classify the given message into multiple categories. 
* data/process_data.py - The python scritp that need to be run to get the final dataset used for modelling
* data/disaster_response db - SQL Database of merged and cleaned CSV files (result of running the process_data.py file)
* models/train_classifier.py - The python script that creates and trains a machine learning model, to classify the messages. The final model is saved as a pickel file. The model accuracy, and classification report is preinted on the terminal
* app/run - The Flask application, that displays data visualizations and gets the predictions for the input message
* app/templates - Contains the html files for the web application
* plots - image folder containing screenshots of final application


To RUN:
* python data/process_data.py data/messages.csv data/categories.csv data/disaster_response.db
  * process_data.py: 
    * input: file path of 2 datasets and database
    * output: stores cleaned SQLite database in specified path
    * function: Merges input csv files of messages and categories,and makes it suitable to be used to train a model
    
* python models/train_classifier.py data/disaster_response.db models/classifier.pkl
  * train_classifier.py: 
    * input: database file path, model file path
    * output: Creates and trains a classifier models and stores it in the specified path
    
* python app/run.py
  * Web application deployed at the specified port number
