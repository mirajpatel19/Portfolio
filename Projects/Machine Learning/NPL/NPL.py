#Import libaries
import pandas as pd
import string
import nltk
#nltk.download_shell()
from nltk.corpus import stopwords

#Load the data
reviews = pd.read_csv('Restaurant_Reviews.tsv', sep='\t')

#Helper Function
def text_process(mess):
    """
    -Remove punctuations by looking at each character
    -Join each character
    -Returns only useful words
    """
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

#Clean the texts
clean_reviews = reviews['Review'].apply(text_process)

#Bag of words model
from sklearn.feature_extraction.text import CountVectorizer

#.....
from sklearn.feature_extraction.text import TfidfTransformer

#Selecting the model
from sklearn.naive_bayes import MultinomialNB


#--------------------The Actual Model--------------------
#Split the data
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(reviews['Review'],reviews['Liked'],test_size=0.3)

#Create the pipeline
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
        ('bow',CountVectorizer(analyzer=text_process)),
        ('tfidf',TfidfTransformer()),
        ('classifier', MultinomialNB())
        ])

#Fit the pipeline
pipeline.fit(X_train,y_train)

#Make predictions with the pipeline
predictions = pipeline.predict(X_test)

#Evaluation the model
from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(y_test,predictions))
print (confusion_matrix(y_test, predictions))