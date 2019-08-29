import tweepy
import csv
import numpy 
import math
import matplotlib.pyplot as plt
from textblob import TextBlob
import pandas
from keras.models import Sequential
from keras.layers import Dense

# This code is based on LP4DS video series from Siraj Ravel
# and his challenge to make a neural network that would predict stock
# market prices based on previous prices and sentiment analysis extracted from Twitter.

consumer_key= 'INSERT YOUR TWITTER API KEYS'
consumer_secret= 'INSERT YOUR TWITTER API KEYS'

access_token='INSERT YOUR TWITTER API KEYS'
access_token_secret='INSERT YOUR TWITTER API KEYS'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# I chose Apple stocks
public_tweets = api.search('#AAPL')

#
sum = 0

for tweet in public_tweets:
    
    # Perform Sentiment Analysis on Tweets
    analysis = TextBlob(tweet.text)
    # Naive algorithm that takes in consideration both polarity and subjectivity of tweets
    sum += analysis.polarity*(1-analysis.subjectivity)

# CSV file read
dataframe = pandas.read_csv('dataset.csv', usecols=[1], engine='python')
dataset = dataframe.values
dataset = dataset.astype('float32')
dataset = dataset[::-1]
last3 = numpy.array([dataset[len(dataset)-3:,0]])

# Building neural network model using Keras, training it, 
# then having it predict the price on the next day. 
# Enter mentioned process if sentiment analysis was 'positive' 

if sum>0:
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size

    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    print(len(train), len(test))
    # convert an array of values into a dataset matrix
    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return numpy.array(dataX), numpy.array(dataY)
    look_back = 3
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # creating and fitting Multilayer Perceptron model
    model = Sequential()
    model.add(Dense(12, input_dim=look_back, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=400, batch_size=2, verbose=2)
    # Estimate model performance
    trainScore = model.evaluate(trainX, trainY, verbose=0)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # Shifting train predictions for plotting purposes (so it can be more easily analyzed)
    trainPredictPlot = numpy.empty_like(dataset)
    trainPredictPlot[:, :] = numpy.nan
    trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
    # Shifting test predictions for plotting
    testPredictPlot = numpy.empty_like(dataset)
    testPredictPlot[:, :] = numpy.nan
    testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
    # Plotting data prices alongside model predictions of train and test set
    plt.plot(dataset, color="blue")
    plt.plot(trainPredictPlot, color="red")
    plt.plot(testPredictPlot, color="yellow")
    plt.show()
    print ("Price predicted for next day is", model.predict(last3))
else:
    print ("Sentiment analysis predicted < 0 ")
