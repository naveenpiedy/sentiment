import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

test_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_toz = PunktSentenceTokenizer(test_text)

to = custom_toz.tokenize(sample_text)

def process_content():
    try:
        for i in to[:100]:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)

            print(tagged)
    except Exception as e:
        print(str(e))

process_content()