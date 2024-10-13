


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import re
from collections import Counter



# from nltk.tokenize import word_tokenize

#load the Excel file of train and print the first few rows of the DataFrame in Python
train_df = pd.read_excel('/content/train_ex1.xlsx')
print(train_df.head())

#load the Excel file of validation and print the first few rows of the DataFrame in Python
val_df = pd.read_excel('/content/val_ex1.xlsx')
print(val_df.head())

import nltk
nltk.download('punkt')
nltk.download('stopwords')
import nltk  # Import nltk inside the function
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
nltk.download('wordnet')

# from collections import Counter
# from autocorrect import Speller

#With this function we have done many tests in order to know which pre-processing processes and in which order will lead to the best results.
#Each time we recorded the results of the accuracy  and error so that we could compare and check which is the best
# ----------------------------------------------------------------
###original
# #Loss: 0.6286246180534363, Accuracy: 0.6944444179534912
# def preprocess_text(text):

#     word_tokens = word_tokenize(text)

#     # Convert to lowercase
#     lower_text = [w.lower() for w in word_tokens]

#     # Join the tokens back into a single string
#     return ' '.join(lower_text)
#     # stop_words = set(stopwords.words('english'))
#     # word_tokens = word_tokenize(text)
#     # filtered_text = [w.lower() for w in word_tokens if w.lower() not in stop_words and w not in string.punctuation]
#     # return ' '.join(filtered_text)
# ----------------------------------------------------------------

#Loss: 0.6790039539337158, Accuracy: 0.6666666865348816
# def preprocess_text(text):
#     # Tokenize the text
#     word_tokens = word_tokenize(text)

#     # Initialize the WordNet Lemmatizer
#     lemmatizer = WordNetLemmatizer()

#     # Lemmatize each word and convert to lowercase
#     lemmatized_text = [lemmatizer.lemmatize(w.lower()) for w in word_tokens]

#     # Join the tokens back into a single string
#     return ' '.join(lemmatized_text)
# ----------------------------------------------------------------

#Loss: 0.5923327207565308, Accuracy: 0.7037037014961243
#Loss: 0.582383394241333, Accuracy: 0.7592592835426331
#Loss: 0.6149447560310364, Accuracy: 0.7222222089767456
# def preprocess_text(text):
#     # Tokenize the text
#     word_tokens = word_tokenize(text)

#     # Remove non-alphabetic characters and convert to lowercase
#     filtered_tokens = [w.lower() for w in word_tokens if w.isalpha()]

#     # Join the tokens back into a single string
#     return ' '.join(filtered_tokens)

# ----------------------------------------------------------------


#Loss: 0.6356053948402405, Accuracy: 0.7314814925193787
#Loss: 0.6211562156677246, Accuracy: 0.7037037014961243
#Loss: 0.5955384373664856, Accuracy: 0.7129629850387573
#Loss: 0.618004322052002, Accuracy: 0.7222222089767456
# def preprocess_text(text):
#     # Tokenize the text
#     word_tokens = word_tokenize(text)

#     # Initialize the WordNet Lemmatizer
#     lemmatizer = WordNetLemmatizer()

#     # Lemmatize each word, remove non-alphabetic characters, and convert to lowercase
#     filtered_tokens = [lemmatizer.lemmatize(w.lower()) for w in word_tokens if w.isalpha()]

#     # Join the tokens back into a single string
#     return ' '.join(filtered_tokens)

# ----------------------------------------------------------------


#Loss: 0.6556088924407959, Accuracy: 0.6388888955116272
#Loss: 0.6333094239234924, Accuracy: 0.6666666865348816
#Loss: 0.6428742408752441, Accuracy: 0.6574074029922485

# def preprocess_text(text):
#     # Tokenize the text
#     word_tokens = word_tokenize(text)

#     # Initialize the Porter Stemmer
#     stemmer = PorterStemmer()

#     # Stem each word, remove non-alphabetic characters, and convert to lowercase
#     filtered_tokens = [stemmer.stem(w.lower()) for w in word_tokens if w.isalpha()]

#     # Join the tokens back into a single string
#     return ' '.join(filtered_tokens)
# ----------------------------------------------------------------

#Loss: 0.6220806241035461, Accuracy: 0.7037037014961243
#Loss: 0.6283233165740967, Accuracy: 0.6944444179534912

# def preprocess_text(text):
#     # Remove non-alphanumeric characters and convert to lowercase
#     filtered_text = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

#     # Tokenize the filtered text
#     word_tokens = word_tokenize(filtered_text)

#     # Join the tokens back into a single string
#     return ' '.join(word_tokens)
# ----------------------------------------------------------------

#Loss: 0.6466468572616577, Accuracy: 0.6666666865348816
#Loss: 0.6637557744979858, Accuracy: 0.6203703880310059

# def preprocess_text(text):
#     # Tokenize the text
#     word_tokens = word_tokenize(text)

#     # Define stopwords and punctuation
#     stop_words = set(stopwords.words('english'))
#     punctuation = set(string.punctuation)

#     # Remove stopwords and punctuation, and filter out non-alphabetic tokens
#     filtered_tokens = [w.lower() for w in word_tokens
#                        if w.lower() not in stop_words and w.lower() not in punctuation and w.isalpha()]

#     # Join the tokens back into a single string
#     return ' '.join(filtered_tokens)

# ----------------------------------------------------------------


#Loss: 0.49832507967948914, Accuracy: 0.7685185074806213
#Loss: 0.5076868534088135, Accuracy: 0.7592592835426331
#Loss: 0.5220251679420471, Accuracy: 0.7592592835426331
#Loss: 0.5085152387619019, Accuracy: 0.7777777910232544
#Loss: 0.5075650811195374, Accuracy: 0.75

# def preprocess_text(text):
#     # Tokenize the text
#     word_tokens = word_tokenize(text)

#     # Convert to lowercase and filter out non-alphabetic tokens
#     filtered_tokens = [w.lower() for w in word_tokens if w.isalpha()]

#     # Calculate word frequencies
#     word_freq = Counter(filtered_tokens)

#     # Determine the top 10 most frequent words
#     top_10_words = [word for word, _ in word_freq.most_common(10)]

#     # Remove the top 10 most frequent words from filtered tokens
#     filtered_tokens = [w for w in filtered_tokens if w not in top_10_words]

#     # Join the tokens back into a single string
#     return ' '.join(filtered_tokens)
# ----------------------------------------------------------------


#Loss: 0.5134583115577698, Accuracy: 0.7685185074806213
#Loss: 0.5196259617805481, Accuracy: 0.7407407164573669
#Loss: 0.5038768649101257, Accuracy: 0.75
#Loss: 0.5178530216217041, Accuracy: 0.75
#Loss: 0.5074611306190491, Accuracy: 0.7685185074806213
#Loss: 0.5089508295059204, Accuracy: 0.7777777910232544
# def preprocess_text(text):
#     # Tokenize the text
#     word_tokens = word_tokenize(text)

#     # Convert to lowercase and filter out non-alphabetic tokens
#     filtered_tokens = [w.lower() for w in word_tokens if w.isalpha()]

#     # Calculate word frequencies
#     word_freq = Counter(filtered_tokens)

#     # Determine the top 10 most frequent words
#     top_10_words = set([word for word, _ in word_freq.most_common(10)])

#     # Determine the 10 most unique words
#     unique_words = set([word for word, count in word_freq.items() if count == 1][:10])

#     # Remove the top 10 most frequent and 10 most unique words from filtered tokens
#     filtered_tokens = [w for w in filtered_tokens if w not in top_10_words and w not in unique_words]

#     # Join the tokens back into a single string
#     return ' '.join(filtered_tokens)


# ----------------------------------------------------------------
#Loss: 0.5273980498313904, Accuracy: 0.7222222089767456
#Loss: 0.5097578763961792, Accuracy: 0.75
#Loss: 0.517579972743988, Accuracy: 0.7314814925193787
#Loss: 0.5181084871292114, Accuracy: 0.7407407164573669
#Loss: 0.5151126384735107, Accuracy: 0.7129629850387573
#Loss: 0.5270452499389648, Accuracy: 0.7222222089767456

# def preprocess_text(text):
#     # Tokenize the text
#     word_tokens = word_tokenize(text)

#     # Initialize the WordNet Lemmatizer
#     lemmatizer = WordNetLemmatizer()

#     # Convert to lowercase and filter out non-alphabetic tokens
#     filtered_tokens = [w.lower() for w in word_tokens if w.isalpha()]

#     # Lemmatize each word
#     lemmatized_tokens = [lemmatizer.lemmatize(w) for w in filtered_tokens]

#     # Calculate word frequencies
#     word_freq = Counter(lemmatized_tokens)

#     # Determine the top 10 most frequent words
#     top_10_words = set([word for word, _ in word_freq.most_common(10)])

#     # Determine the 10 most unique words
#     unique_words = set([word for word, count in word_freq.items() if count == 1][:10])

#     # Remove the top 10 most frequent and 10 most unique words from filtered tokens
#     filtered_tokens = [w for w in lemmatized_tokens if w not in top_10_words and w not in unique_words]

#     # Join the tokens back into a single string
#     return ' '.join(filtered_tokens)


# ----------------------------------------------------------------
#TheRealBest
#Loss: 0.5205858945846558, Accuracy: 0.7685185074806213
#Loss: 0.5167315006256104, Accuracy: 0.7407407164573669
#Loss: 0.5061954855918884, Accuracy: 0.7777777910232544
#Loss: 0.5157479643821716, Accuracy: 0.75
#Loss: 0.5172320008277893, Accuracy: 0.7777777910232544
#Loss: 0.5161720514297485, Accuracy: 0.7592592835426331
#Loss: 0.5295161008834839, Accuracy: 0.75
#Loss: 0.5150034427642822, Accuracy: 0.7870370149612427
#Loss: 0.5040651559829712, Accuracy: 0.8055555820465088



def preprocess_text(text):
    # Tokenize the text
    word_tokens = word_tokenize(text)

    # Initialize the Porter Stemmer
    stemmer = PorterStemmer()

    # Convert to lowercase and filter out non-alphabetic tokens
    filtered_tokens = [w.lower() for w in word_tokens if w.isalpha()]

    # Stem each word
    stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]

    # Calculate word frequencies
    word_freq = Counter(stemmed_tokens)

    # Determine the top 10 most frequent words
    top_10_words = set([word for word, _ in word_freq.most_common(10)])

    # Determine the 10 most unique words
    unique_words = set([word for word, count in word_freq.items() if count == 1][:10])

    # Remove the top 10 most frequent and 10 most unique words from filtered tokens
    filtered_tokens = [w for w in stemmed_tokens if w not in top_10_words and w not in unique_words]

    # Join the tokens back into a single string
    return ' '.join(filtered_tokens)
# ----------------------------------------------------------------
#Loss: 0.5377179384231567, Accuracy: 0.7777777910232544
#Loss: 0.5142583250999451, Accuracy: 0.7592592835426331
#Loss: 0.5142452120780945, Accuracy: 0.7685185074806213
#Loss: 0.5331658124923706, Accuracy: 0.7592592835426331
#Loss: 0.5222521424293518, Accuracy: 0.7685185074806213
#Loss: 0.5283617377281189, Accuracy: 0.7592592835426331
# def preprocess_text(text):
#     # Tokenize the text
#     word_tokens = word_tokenize(text)

#     # Initialize the Porter Stemmer and WordNet Lemmatizer
#     stemmer = PorterStemmer()
#     lemmatizer = WordNetLemmatizer()

#     # Convert to lowercase and filter out non-alphabetic tokens
#     filtered_tokens = [w.lower() for w in word_tokens if w.isalpha()]

#     # Stem and Lemmatize each word
#     stemmed_tokens = [stemmer.stem(lemmatizer.lemmatize(w)) for w in filtered_tokens]

#     # Calculate word frequencies
#     word_freq = Counter(stemmed_tokens)

#     # Determine the top 10 most frequent words
#     top_10_words = set([word for word, _ in word_freq.most_common(10)])

#     # Determine the 10 most unique words
#     unique_words = set([word for word, count in word_freq.items() if count == 1][:10])

#     # Remove the top 10 most frequent and 10 most unique words from filtered tokens
#     filtered_tokens = [w for w in stemmed_tokens if w not in top_10_words and w not in unique_words]

#     # Join the tokens back into a single string
#     return ' '.join(filtered_tokens)
# from collections import Counter
# from spellchecker import SpellChecker

# ----------------------------------------------------------------

#Loss: 0.5219728946685791, Accuracy: 0.7314814925193787
#Loss: 0.5199993848800659, Accuracy: 0.7685185074806213
#Loss: 0.5292153358459473, Accuracy: 0.7314814925193787
#Loss: 0.5220841765403748, Accuracy: 0.75
#Loss: 0.515212893486023, Accuracy: 0.7685185074806213
#Loss: 0.5210123658180237, Accuracy: 0.7777777910232544


# def preprocess_text(text):
#     # Tokenize the text
#     word_tokens = word_tokenize(text)

#     # Initialize the Porter Stemmer
#     stemmer = PorterStemmer()

#     # Initialize the spell checker
#     spell = Speller()

#     # Correct spelling of words
#     corrected_tokens = [spell(w) for w in word_tokens]

#     # Convert to lowercase and filter out non-alphabetic tokens
#     filtered_tokens = [w.lower() for w in corrected_tokens if w.isalpha()]

#     # Stem each word
#     stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]

#     # Calculate word frequencies
#     word_freq = Counter(stemmed_tokens)

#     # Determine the top 10 most frequent words
#     top_10_words = set([word for word, _ in word_freq.most_common(10)])

#     # Determine the 10 most unique words
#     unique_words = set([word for word, count in word_freq.items() if count == 1][:10])

#     # Remove the top 10 most frequent and 10 most unique words from filtered tokens
#     filtered_tokens = [w for w in stemmed_tokens if w not in top_10_words and w not in unique_words]

#     # Join the tokens back into a single string
#     return ' '.join(filtered_tokens)
# ----------------------------------------------------------------


#activate preprocess_text on train and val
train_df['text'] = train_df['text'].apply(preprocess_text)
val_df['text'] = val_df['text'].apply(preprocess_text)

print(train_df.head())
print(val_df.head())

#do tf-idf vectorization
all_text = pd.concat([train_df['text'], val_df['text']])

vectorizer = TfidfVectorizer()
vectorizer.fit(all_text)

X_train = vectorizer.transform(train_df['text']).toarray()
X_test = vectorizer.transform(val_df['text']).toarray()
y_train = train_df['label'].to_numpy()
y_test = val_df['label'].to_numpy()

#Building the neural network, we played with the parameters until we reached an optimal result.
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

#compile the NN
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Training the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

#We printed the error and the accuracy to know every time what the best run was.
# The goal was to increase the accuracy and decrease the error.
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Loss: {loss}, Accuracy: {accuracy}")

##load the Excel file of test and print the first few rows of the DataFrame in Python

test_df = pd.read_excel('/content/test_ex1.xlsx')
print(test_df.head())

#predicted the target - label of test
test_df['text'] = test_df['text'].apply(preprocess_text)
X_test_df = vectorizer.transform(test_df['text']).toarray()
predictions = model.predict(X_test_df)
print(predictions)
predicted_classes = (predictions > 0.5).astype(int)
test_df['predicted_label'] = predicted_classes
print(X_test_df)

#Loading the results to a csv file
test_df[['id', 'predicted_label']].to_csv('predicted_results.csv', index=False, header=['id', 'label'])