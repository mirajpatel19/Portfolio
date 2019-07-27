#Import libaries
import pandas as pd
import nltk
#nltk.download_shell()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from textblob import TextBlob

#Load the data
reviews = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')

#Helper Function
def process_text(sentences):
    """
    -Split the sentence and remove punctuations by looking at each character
    -Remove the common words(the, a, is) and apply stemming(removes same words(liked, like))
    -Join the words back to form the original sentance 
    -Add the the full sentance to the corpus, which is an array
    -Convert corput to the list and return the clean version
    """
    corpus = []
    for sentence in sentences:
        texts = re.sub('[^a-zA-Z]',' ', sentence).split()
        join_words = ' '.join(texts)
        corpus.append(join_words)
    return pd.DataFrame(corpus, columns=['Reviews'])
    
#Clean the sentences       
clean_list = process_text(reviews['Review'])

print (type(clean_list))

clean_list['Polarity'] = clean_list['Reviews'].apply(lambda x: TextBlob(x).sentiment.polarity)
clean_list['Subjectivity'] = clean_list['Reviews'].apply(lambda x: TextBlob(x).sentiment.subjectivity)

