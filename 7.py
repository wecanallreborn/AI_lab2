import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names

def gender_features(word, n):
    return {'feature': word[-n:].lower()}

data = ([(n, 'male') for n in names.words('male.txt')] +
        [(n, 'female') for n in names.words('female.txt')])
random.seed(7)
random.shuffle(data)

my_names = ['Conan', 'Sherlock', 'Mary', 'Rita', 'Isaac', 'Coraline', 'Magdaline', 'Stephan', 'Martin', 'Cordelia']
split = int(0.8 * len(data))

for n in range(1, 6):
    feats = [(gender_features(name, n), g) for name, g in data]
    train_set, test_set = feats[:split], feats[split:]
    clf = NaiveBayesClassifier.train(train_set)
    print(f"N={n}: accuracy = {nltk_accuracy(clf, test_set):.3f}")
    for nm in my_names:
        print(f"   {nm:<10} {clf.classify(gender_features(nm, n))}")
