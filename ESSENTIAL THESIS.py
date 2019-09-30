

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
#import requests
#from bs4 import BeautifulSoup

import spacy

#spaCy preprocessing parameters

preprocessing = {

'remove_quotes': True,

'replace_curly_quotes': True,

'strip_punctuation': True,

'POS': ['ADJ', 'ADV', 'INTJ', 'NOUN', 'VERB'],

'lowercase': True,

'letters_only': True,

'min_occurrences': 10

}

nlp = spacy.load('en', disable=['parser', 'ner'])

#preprocessing of text with spaCy

def preprocess(text):
    features = []
    previous_lemma = False
    for token in nlp(text):
        valid_pos = token.pos_ in preprocessing['POS']
        lemma = token.lemma_
        if valid_pos and (lemma.isalpha() or not preprocessing['letters_only']):
            if preprocessing['lowercase']:
                lemma = lemma.lower()
            features.append(lemma)
            if previous_lemma:
                features.append(' '.join([previous_lemma, lemma]))
            previous_lemma = lemma
        else:
            previous_lemma = False
    return features


# create lists of texts and labels
texts = []
labels = []

for filename in ["MAINstream.txt", "HYPERpart.txt"]:
    TX = open(filename, encoding='utf-8').read().split("\n")[:-1]
    n_texts = len(TX)
#   assign label name, create list of labels
    labels = labels + [filename[:-4]] * n_texts
#   create list of texts
    texts = texts + TX
    

    
# create test and training data sets
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, random_state=0)


# Extracting features using either a count or TFIDF vectorizer

vect = CountVectorizer(analyzer=preprocess,
                       min_df=preprocessing['min_occurrences'])
                       #stop_words="english", #token_pattern = "[a-zA-Z]{2,}",                       
                       #ngram_range = (1, 2),
                       #max_df=0.99,
                       #min_df=0.1)

#vect = TfidfVectorizer(stop_words='english', 
#                       token_pattern = "[a-zA-Z]{2,}",
#                       ngram_range = (1, 2),
#                       max_df=0.99, 
#                       min_df=0.1) 




# Fitting model

clfrNB = MultinomialNB()


#create pipeline

pipe = Pipeline([('vectorize', vect),
                 ('classify', clfrNB)])
    
pipe.fit(texts_train, labels_train)

predicted_train = pipe.predict(texts_train)
predicted_test = pipe.predict(texts_test)



#Get accuracy score
score = pipe.score(labels_test, predicted_test)
print("accuracy score is", score)

print(confusion_matrix(labels_test, predicted_test))
print(classification_report(labels_test, predicted_test))

five_fold_cross = cross_validate(pipe, texts_train, labels_train,
                              cv = 5)

#Most informative features function

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))

show_most_informative_features(vect, clfrNB)


# Use exogenous file as an extra test. This is a "fake news" file about the death of Barbara Bush


trial_text = open("BABS_BUSH_FAKE.txt", encoding='utf-8').read()

print(pipe.predict([trial_text]))

#print 5-fold cross validation
print(five_fold_cross)


