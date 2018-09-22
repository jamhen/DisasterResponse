# Disaster Response Pipeline Project
### Introduction
In this Project, we would be analyzing message data for disaster response. We perform ETL (Extract Transform Load) on the data then build a pipeline to process the data, then use a Machine Learning algorithm, supervised learning model, to predict the results. The messages used in this dataset were taken from [Figure Eight](https://www.figure-eight.com/). 
The project includes a web app that would help emergency workers can input a new message concerning a disaster and get classifications results for different responses needed based on the message.

### Motivation
Following a disaster, we can get millions of communications either direct, by social media or from the news, right at the time when disaster response organizations have the least capacity to filter and pull out the messages which are most important. Also, not every message might be relevant to disaster response professionals since different organizations take care of different parts of the disaster problem, some can take care of food, some water, medical aid; and so on.

Thus supervised machine learning based approaches would be of good assistance to accurately search for keywords in a message and predict which disaster response is needed. Another drive for this project is to see how accurate we can use machine learning models that can help us to respond to future disasters.

### Files
* disaster_messages.csv: File containing messages data from different from different sources on disasters
* disaster_categories.csv: File containing labels of disaster response categories.
* process_data.py: File to prepare the the data contained in disaster_messages.csv and disaster_categories.
* train_classifier.py: File to train a model to classify messages.
* run.py: Run web app to analyse message and display predictions.
* master.html: Web file to display results
* go.html: Web file to display results.
* results.png: Image of web page.

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
