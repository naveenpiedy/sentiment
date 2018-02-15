from nltk.stem import PorterStemmer
from nltk.tokenize import  word_tokenize

ps = PorterStemmer()

example_words = ["love", "loving", "loved", "lovingly"]

print([ps.stem(w) for w in example_words])