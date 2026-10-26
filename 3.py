from nltk.stem import WordNetLemmatizer

words = "Ms.Smith's dog was barking for a long time while i was trying to fall asleep ".split()

print([WordNetLemmatizer().lemmatize(i, pos='n') for i in words])
print([WordNetLemmatizer().lemmatize(i, pos='v') for i in words])
