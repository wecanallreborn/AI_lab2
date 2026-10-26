from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

def extract_features(words):
    return {w: True for w in words}

pos_ids, neg_ids = movie_reviews.fileids('pos'), movie_reviews.fileids('neg')
f_pos = [(extract_features(movie_reviews.words(fileids=[f])), 'Positive') for f in pos_ids]
f_neg = [(extract_features(movie_reviews.words(fileids=[f])), 'Negative') for f in neg_ids]

num_pos, num_neg = int(0.8 * len(f_pos)), int(0.8 * len(f_neg))
f_train = f_pos[:num_pos] + f_neg[:num_neg]
f_test = f_pos[num_pos:] + f_neg[num_neg:]
print("Train:", len(f_train), "Test:", len(f_test))

clf = NaiveBayesClassifier.train(f_train)
print("Accuracy:", round(nltk_accuracy(clf, f_test), 3))
print("Top-20 informative words:")
clf.show_most_informative_features(20)

reviews = ["The film was absolutely fantastic, with brilliant acting.",
           "A boring and terrible movie, a complete waste of time.",
           "I loved the story and the wonderful soundtrack.",
           "The plot was dull and the characters were awful.",
           "It was an okay film, nothing special.",
           "Spooky and creapy film. But horror is my cap of tea."]
for r in reviews:
    probs = clf.prob_classify(extract_features(r.split()))
    print(f"{probs.max():<9} P(pos)={probs.prob('Positive'):.2f} P(neg)={probs.prob('Negative'):.2f} | {r}")
