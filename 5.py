import numpy as np
from nltk.corpus import brown
from sklearn.feature_extraction.text import CountVectorizer

def chunker(input_data, N):
    words = input_data.split(' ')
    return [' '.join(words[i:i + N]) for i in range(0, len(words), N)]


text = ' '.join(brown.words()[:12000])
chunks = chunker(text, 700)
items = [{'index': i, 'text': c} for i, c in enumerate(chunks)]

vec = CountVectorizer(min_df=5, max_df=0.95)
dtm = vec.fit_transform([it['text'] for it in items])

vocab = np.array(vec.get_feature_names_out())
print("Форма матрицы:", dtm.shape)
print("Слов в словаре:", len(vocab))
top = dtm.sum(axis=0).A1.argsort()[::-1][:10]
print("Топ-10 слов:", vocab[top])
print(dtm.toarray()[:, top])