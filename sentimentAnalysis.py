from textblob import TextBlob
import csv

f = open("Sentiment_Republic_Tweets.csv", "w")
my_writer = csv.writer(f)

with open("Republican_Tweets.csv", "r") as csvfile:
    my_reader = csv.reader(csvfile)
    for row in my_reader:
        text = row[2].decode("utf-8")
        testimonial = TextBlob(text)

        if testimonial.sentiment.polarity > 0:
            sentiment = "positive"
        elif testimonial.sentiment.polarity < 0:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        my_writer.writerow([row[0], row[1], row[2], row[3], sentiment])

f.close()


f = open("Sentiment_Democrat_Tweets.csv", "w")
my_writer = csv.writer(f)

with open("Democrats_Tweets.csv", "r") as csvfile:
    my_reader = csv.reader(csvfile)
    for row in my_reader:
        text = row[2].decode("utf-8")
        testimonial = TextBlob(text)

        if testimonial.sentiment.polarity > 0:
            sentiment = "positive"
        elif testimonial.sentiment.polarity < 0:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        my_writer.writerow([row[0], row[1], row[2], row[3], sentiment])

f.close()

