# %%
import pandas as pd
import numpy as np
import os, re, datetime, json, pickle
from tensorflow import keras
from keras.utils import pad_sequences, plot_model
from keras.preprocessing.text import Tokenizer
from keras import Sequential
from keras.layers import LSTM, Dense, Embedding, Dropout
from keras.callbacks import TensorBoard
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = os.path.join(os.getcwd(), 'dataset')

# %% 1. Data loading
df = pd.read_csv(os.path.join(DATASET_PATH, 'True.csv'))
# df = df.loc[:, ['text', 'subject']]

# %% 2. Data inspection
df.info()
df.describe()

df.isna().sum() # no NaN values
df.duplicated().sum() # 206 rows of duplication for all column

print(df['text'][8]) # Links, symbols and Twitter username should be remove if possible

# %% 3. Data cleaning
# Define features and labels
features = df['text'] 
labels = df['subject']

# Remove unnecessary information in the data and normalize the text
def clean_text(text):
    filter = '(^[\w ,./\(\)]*\([\w]+\))|(@[\w]*)|(\[[\w ]+\])|(bit.ly/[\w]*)|([^a-zA-Z])'
    return re.sub(filter, ' ', text).lower()

features = features.apply(clean_text)

# Remove duplicate
df_drop = pd.concat([features,labels], axis=1)
df_drop = df_drop.drop_duplicates()
df_drop.duplicated().sum()

print(df_drop['text'][8])

# Number of words
word_count = [len(text.split()) for text in features]

np.median(word_count)
np.mean(word_count)

# %% 4. Features selection
# Define the labels
features = df_drop['text'] 
labels = df_drop['subject']

# %% 5. Data pre-preprocessing
# Tokenization
vocab_num = 5000
oov_token = '<OOV>'

tokenizer = Tokenizer(num_words=vocab_num, oov_token=oov_token)
tokenizer.fit_on_texts(features)

train_sequences = tokenizer.texts_to_sequences(features)

# Padding + truncating
train_sequences = pad_sequences(train_sequences, maxlen=400, padding='post', truncating='post')

# Expand the dimension of the labels and features into 2d array
train_sequences = np.expand_dims(train_sequences, -1)
train_labels = np.expand_dims(labels, -1)

# Pre-processing for the labels - Encode the label with OneHotEncoding
ohe = OneHotEncoder(sparse=False)
train_labels = ohe.fit_transform(train_labels)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(train_sequences, train_labels, random_state=123)

# %% Model development
embedding_dim = 64

model = Sequential()
model.add(Embedding(vocab_num, embedding_dim))
model.add(LSTM(embedding_dim, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(y_train.shape[1], activation='softmax'))

model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)

# Model compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# Define callbacks
LOG_DIR = os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = TensorBoard(log_dir=LOG_DIR)

# Model training
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks=tb)

# %% Model evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)

# Evaluate prediction
print('Confusion Matrix:\n', confusion_matrix(y_pred, y_true))
print('Classification Report:\n', classification_report(y_pred, y_true))

# %% Model saving
# Save tokenizer
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Save OHE
with open('ohe.pkl', 'wb') as f:
    pickle.dump(ohe, f)

# Save model
model.save('text-classification.h5')

# %%
