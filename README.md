# Disaster response pipeline

Project summary:

How to run:
* python scripts:
  * elt_script : 
    *input: file path of 2 datasets and database
    *output: stores cleaned SQLite database in specified path
    *function: Merges input csv files of messages and categories,and makes it suitable to be used to train a model
  * train_classifier.py: 
    *input: database file path, model file path
    *output: Creates and trains a classifier models and stores it in the specified path
    
* web app:
  *python app/run.py

Files:

* ELT Pipeline Preperation notebook - Documents the data preperation process from CSV files, messages and categories
* disaster_response db - SQL Database of merged and cleaned CSV files
