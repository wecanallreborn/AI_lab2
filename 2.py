from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import SnowballStemmer

words = "мама мыла раму с мылом".split()

print([PorterStemmer().stem(i) for i in words])
print([LancasterStemmer().stem(i) for i in words])
print([SnowballStemmer('russian').stem(i) for i in words])

words = "Ms.Smith's dog was barking for a long time while i was trying to fall asleep ".split()

print([PorterStemmer().stem(i) for i in words])
print([LancasterStemmer().stem(i) for i in words])
print([SnowballStemmer('english').stem(i) for i in words])