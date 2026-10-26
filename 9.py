from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim import models, corpora

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer('english')
stop = set(stopwords.words('english'))

with open('data.txt', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]

tokens = []
for line in lines:
    ws = [w for w in tokenizer.tokenize(line.lower()) if w not in stop]
    tokens.append([stemmer.stem(w) for w in ws])

dict_tokens = corpora.Dictionary(tokens)
doc_term_mat = [dict_tokens.doc2bow(t) for t in tokens]

num_topics = 2
ldamodel = models.ldamodel.LdaModel(doc_term_mat, num_topics=num_topics,
                                    id2word=dict_tokens, passes=25, random_state=1)
for idx, topic in ldamodel.print_topics(num_topics=num_topics, num_words=5):
    print(f"Topic {idx}: {topic}")
