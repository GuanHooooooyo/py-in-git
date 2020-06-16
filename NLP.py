import spacy
import en_qai_sm
from spacy.matcher import PhraseMatcher
import pandas as pd
from spacy.util import minibatch

#nlp = en_qai_sm.load()

#doc = nlp("Tea is healthy and calming, don't you think?")
#
# for token in doc:
#     print(token)

# print(f"Token \t\tLemma \t\tStopword".format('Token', 'Lemma', 'Stopword'))
# print("-"*40)
# for token in doc:
#     print(f"{str(token)}\t\t{token.lemma_}\t\t{token.is_stop}")
#
# matcher = PhraseMatcher(nlp.vocab, attr='LOWER')
#
# terms = ['Galaxy Note', 'iPhone 11', 'iPhone XS', 'Google Pixel']
# patterns = [nlp(text) for text in terms]
# matcher.add("TerminologyList", None, *patterns)
#
# text_doc = nlp("Glowing review overall, and some really interesting side-by-side "
#                "photography tests pitting the iPhone 11 Pro against the "
#                "Galaxy Note 10 Plus and some test is good in iPhone 11 "
#                "last year’s iPhone XS and Google Pixel 3.")
# matches = matcher(text_doc)
# print(matches)
# match_id, start, end = matches[0]
# print(nlp.vocab.strings[match_id], text_doc[start:end])

df = pd.read_csv('D:\py in git\spam.csv')
df.head(5)
print(df)
nlp = spacy.blank("en")
print('success')
textcat = nlp.create_pipe(
    "textcat",config={
    "exclusive_classes": True,
    "architecture": "bow"})

# Add the TextCategorizer to the empty model
nlp.add_pipe(textcat)
textcat.add_label('ham')
textcat.add_label('spam')
print(textcat)
train_texts = df['text'].values
train_labels = [{'cats': {'ham': label == 'ham',
                          'spam': label == 'spam'}}
                for label in df['type']]
train_data = list(zip(train_texts, train_labels))
print(train_data[:3])


import random
random.seed(1)
spacy.util.fix_random_seed(1)
optimizer = nlp.begin_training()

losses = {}
for epoch in range(10):
    random.shuffle(train_data)
    # Create the batch generator with batch size = 8
    batches = minibatch(train_data, size=8)
    # Iterate through minibatches
    for batch in batches:
        # Each batch is a list of (text, label) but we need to
        # send separate lists for texts and labels to update().
        # This is a quick way to split a list of tuples into lists
        texts, labels = zip(*batch)
        nlp.update(texts, labels, sgd=optimizer, losses=losses)
    print(losses)
texts = ["Are you ready for the tea party????? It's gonna be wild",
         "URGENT Reply to this message for GUARANTEED FREE TEA"]
docs = [nlp.tokenizer(text) for text in texts]

# Use textcat to get the scores for each doc
textcat = nlp.get_pipe('textcat')
scores, _ = textcat.predict(docs)

print(scores)

# From the scores, find the label with the highest score/probability
predicted_labels = scores.argmax(axis=1)
print([textcat.labels[label] for label in predicted_labels])