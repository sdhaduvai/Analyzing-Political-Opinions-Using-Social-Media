from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import unicodedata

stopwords = set(STOPWORDS)
import csv

stopwords.add('https')
stopwords.add('co')

def show_wordcloud(data, title=None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1  # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.savefig("Democrats_Word_Cloud.png")
    plt.show()


if __name__ == "__main__":
    with open("Democrats_Tweets.csv", "r") as f:
        my_csv = csv.reader(f)
        lines = []
        for row in my_csv:
            try:
                p = unicodedata.normalize('NFKD', unicode(row[2])).encode('ascii', 'ignore')
                lines.append(p)
            except Exception as e:
                print(e)
                print(p)
    text = "".join(lines)

    show_wordcloud(text)

    