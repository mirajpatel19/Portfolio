#Import libaries
import pandas as pd
import nltk
#nltk.download_shell()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


#Load the data
reviews = pd.read_csv('Data/Restaurant_Reviews.tsv', sep='\t')

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
    ps = PorterStemmer()
    for sentence in sentences:
        texts = re.sub('[^a-zA-Z]',' ', sentence).split()
        texts = [ps.stem(word) for word in texts if not word.lower() in set(stopwords.words('english'))]
        join_words = ' '.join(texts)
        corpus.append(join_words)
    corpus_tolist = list(corpus)
    return corpus_tolist
        
#Clean the sentences       
clean_list = process_text(reviews['Review'])
 
#Bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
cv_bag_of_words = cv.fit_transform(clean_list)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
bag_of_words = tfidf.fit_transform(cv_bag_of_words)


X = pd.DataFrame(bag_of_words.toarray(), columns=cv.get_feature_names())
y = reviews['Liked']

#Split the data
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#Selecting the model
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()

#Fit the model
mnb.fit(X_train,y_train)

#Make predictions with the pipeline
predictions = mnb.predict(X_test)

#Evaluation the model
from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(y_test, predictions))
print (confusion_matrix(y_test, predictions))

