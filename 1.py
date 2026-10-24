from nltk.tokenize import sent_tokenize, word_tokenize, WordPunctTokenizer

text = "В Санкт-Петербруге опять дождь. Он был, есть и будет есть."
print("Исходный текст:", text)
print("Предложения:", sent_tokenize(text))
print("Слова:", word_tokenize(text))
print("Слова и пунктуация:", WordPunctTokenizer().tokenize(text))


