from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

category_map = {'talk.politics.mideast': 'Politics', 'rec.motorcycles': 'Motorcycles',
                'comp.graphics': 'Graphics', 'sci.space': 'Space', 'sci.electronics': 'Electronics'}
cats = list(category_map.keys())

train = fetch_20newsgroups(subset='train', categories=cats, shuffle=True, random_state=5)
cv = CountVectorizer()
train_counts = cv.fit_transform(train.data)
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_counts)

tests = ["The rocket launch to the Moon was delayed by the weather.",
         "My new Harley engine makes a strange noise at high speed.",
         "The parliament discussed peace negotiations in the Middle East.",
         "This GPU renders three-dimensional images very fast.",
         "I soldered a resistor and a capacitor onto the circuit board."]

clf = MultinomialNB().fit(train_tfidf, train.target)
test_tfidf = tfidf.transform(cv.transform(tests))
pred = clf.predict(test_tfidf)
for s, p in zip(tests, pred):
    print(f"{train.target_names[p]:<24} {s}")
