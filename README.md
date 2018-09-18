# Disaster Response Pipeline Project
### Introduction
In this Project we would be analysing message data for disaster reponse. We perform ETL (Extract transform Load) on the data then build a pipline to process the data, then use a Machine Learning algorithm, supervised learning model, to predict the results. The messages used in this data set where taken from [Figure Eight](https://www.figure-eight.com/).

### Motivation
The project includes a web app that would help emergency workers can input new disaster messages and get classifications results on different categories of the messages that will help them respond to desisaster messages.

### Files
* disaster_messages.csv: File containing messages data from different from different sources on disasters
* disaster_categories.csv: File containing labels of disaster response categories.
* process_data.py: File to prepare the the data contained in disaster_messages.csv and disaster_categories.
* DisasterResponse.db: Database file for processed data.
* train_classifier.py: File to train a model to classify messages.
* classifier.plk: Saved model.
* run.py: Run web app to analyse message and display predictions.
* master.html: Web file to display results
* go.html: Web file to display results.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ on your browser

4. Insert a message.

### License
General Public Licence v3.0
### Useful Links
1. [Ploty Bar Charts](https://plot.ly/python/bar-charts/)
2. [Sci-kit Learn MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html)
3. [Figure Eight](https://www.figure-eight.com/data-for-everyone)
