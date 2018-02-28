import nltk
import random
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.corpus import movie_reviews

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from nltk.classify import ClassifierI
from statistics import mode

from nltk.tokenize import word_tokenize


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers= classifiers

    def classify(self, features):
        votes =  []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            return mode(votes)

    def confidence(self, features):
        votes =[]
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        return choice_votes/len(votes)

short_pos = open("short_reviews/positive.txt", 'r').read()
short_neg = open("short_reviews/negative.txt", 'r').read()

all_words = []
documents =[]

#  j is adject, r is adverb, and v is verb
#allowed_word_types = ["J","R","V"]
allowed_word_types = ["J"]

for r in short_pos.split('\n'):
    documents.append((r,'pos'))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

for r in short_neg.split('\n'):
    documents.append((r,'neg'))
    words = word_tokenize(r)
    pos = nltk.pos_tag(words)
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())

save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(documents, save_documents)
save_documents.close()


short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)


all_words = nltk.FreqDist(all_words)

words_features = list(all_words.keys())[:7000]

save_word_features = open("pickled_algos/word_features5k.pickle","wb")
pickle.dump(words_features, save_word_features)
save_word_features.close()


def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in words_features:
        features[w] =(w in words)

    return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

random.shuffle(featuresets)
print(len(featuresets))

training_set = featuresets[:10000]
testing_set = featuresets[10000:]



classifier = nltk.NaiveBayesClassifier.train(training_set)
print("NBA Acc: "+ str((nltk.classify.accuracy(classifier, testing_set))*100))

classifier.show_most_informative_features(15)

save_classifier = open("pickled_algos/originalnaivebayes5k.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()



MNB_classifer = SklearnClassifier(MultinomialNB())
MNB_classifer.train(training_set)
print("MNB Acc: "+ str((nltk.classify.accuracy(MNB_classifer, testing_set))*100))

save_classifier = open("pickled_algos/MNB_classifier5k.pickle","wb")
pickle.dump(MNB_classifer, save_classifier)
save_classifier.close()

BernoulliNB_classifer = SklearnClassifier(BernoulliNB())
BernoulliNB_classifer.train(training_set)
print("BernoulliNB Acc: "+ str((nltk.classify.accuracy(BernoulliNB_classifer, testing_set))*100))

save_classifier = open("pickled_algos/BernoulliNB_classifier5k.pickle","wb")
pickle.dump(BernoulliNB_classifer, save_classifier)
save_classifier.close()


# SVC_classifer = SklearnClassifier(SVC())
# SVC_classifer.train(training_set)
# print("SVC: "+ str((nltk.classify.accuracy(SVC_classifer, testing_set))*100))

LinearSVC_classifer = SklearnClassifier(LinearSVC())
LinearSVC_classifer.train(training_set)
print("LinearSVC: "+ str((nltk.classify.accuracy(LinearSVC_classifer, testing_set))*100))

save_classifier = open("pickled_algos/LinearSVC_classifier5k.pickle","wb")
pickle.dump(LinearSVC_classifer, save_classifier)
save_classifier.close()

# NuSVC_classifer = SklearnClassifier(NuSVC())
# NuSVC_classifer.train(training_set)
# print("NuSvc: "+ str((nltk.classify.accuracy(NuSVC_classifer, testing_set))*100))

LogisticRegression_classifer = SklearnClassifier(LogisticRegression())
LogisticRegression_classifer.train(training_set)
print("LogisticRegression Acc: "+ str((nltk.classify.accuracy(LogisticRegression_classifer, testing_set))*100))

save_classifier = open("pickled_algos/LogisticRegression_classifier5k.pickle","wb")
pickle.dump(LogisticRegression_classifer, save_classifier)
save_classifier.close()

SGDClassifier_classifer = SklearnClassifier(SGDClassifier())
SGDClassifier_classifer.train(training_set)
print("SGDC Acc: "+ str((nltk.classify.accuracy(SGDClassifier_classifer, testing_set))*100))

save_classifier = open("pickled_algos/SGDC_classifier5k.pickle","wb")
pickle.dump(SGDClassifier_classifer, save_classifier)
save_classifier.close()

voted_classifier = VoteClassifier( MNB_classifer, BernoulliNB_classifer, LinearSVC_classifer, LogisticRegression_classifer, SGDClassifier_classifer)

print("voted Acc: "+ str((nltk.classify.accuracy(voted_classifier, testing_set))*100))

print("Classification: "+ str(voted_classifier.classify(testing_set[0][0])), "Confidence : "+str(voted_classifier.confidence(testing_set[0][0])*100))
