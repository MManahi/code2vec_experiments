import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from matplotlib import pyplot

font = {'weight': 'bold', 'size': 20}
plt.rc('font', **font)

dataframe = pd.read_json('data/source_code.json', lines=True)
print(dataframe.head(5))
print('{:,}'.format(len(dataframe)))

#print(dataframe.correct.value_counts())

def remove_comments(text):
    return re.sub(re.compile('\\.*?\n'), '', text)
def get_docs_and_labels(df):
    _docs = []
    _labels = []
    for index in df.index:
        code = remove_comments(
            df.at[index, 'source']
        )
        _docs.append(code)
        label = int(df.at[index, 'submission_id'])
        _labels.append(label)
    return _docs, _labels

docs, labels = get_docs_and_labels(dataframe)
print('{:,}'.format(len(docs)))


NUM_WORDS = 2000

def get_tokenizer():
    return Tokenizer(num_words=NUM_WORDS, 
                     filters='\t\n', 
                     lower=True, 
                     split=' ', 
                     char_level=False)

word_t = get_tokenizer()
word_t.fit_on_texts(docs)
print(word_t.word_counts['if']) # count word (if) in the source code submissions
print('Number docs: {:,}'.format(word_t.document_count)) # total num of submission
print(word_t.word_index['if'])
print(word_t.word_docs['if'])


sequences = word_t.texts_to_sequences(docs)
print(sequences[0])
len_seqs = [len(s) for s in sequences]
np.mean(len_seqs), np.std(len_seqs), np.max(len_seqs)
MAX_LENGTH = 50



id_to_word = { v: k for k, v in word_t.word_index.items() }
print(id_to_word[1])
print([id_to_word[index] for index in sequences[0]])

padded_docs = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')
print(padded_docs[0])

def f1(y_true, y_pred):
    
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def get_model():

    # define the model
    model = Sequential()
    model.add(Embedding(NUM_WORDS, 100, input_length=MAX_LENGTH))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile the model
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['acc', f1])
    # summarize the model
    print(model.summary())
    return model

model = get_model()

X_train, X_test, y_train, y_test = train_test_split(padded_docs, labels, test_size=0.2, random_state=0)
model.fit(X_train, 
          y_train,
          epochs=10,
          validation_split=0.2)
word_loss, word_accuracy, word_f1 = model.evaluate(X_test, y_test, verbose=1)
print('Accuracy: %f, F1: %f' % (word_accuracy * 100, word_f1 * 100))

word_score = {
    'accuracy': word_accuracy,
    'F1': word_f1,
}
embeddings_scores = { 'Word': word_score }
def get_embeddings(model):

    # Embedding Layer
    embedding_layer = model.layers[0]
    embeddings = embedding_layer.get_weights()[0]
    print('Embedding Layer shape:', embeddings.shape)
    
    return embeddings
embeddings = get_embeddings(model)


def get_pca(embeddings):

    # PCA
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(embeddings)
    print('PCA explained variance ratio:', pca.explained_variance_ratio_, 'Total:', sum(pca.explained_variance_ratio_))
    return principal_components
pca = get_pca(embeddings)

def get_top_words(tokenizer, N=50):
    
    return [word for word, occurrences in sorted(tokenizer.word_counts.items(), key=lambda t: t[1], reverse=True)[:N]]

top_words = get_top_words(word_t)

def plot_embeddings(low_dim_embs, id_to_word, top_words, figsize=(8, 8)):

    plt.figure(figsize=figsize, dpi=100)
    ax = plt.axes()
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_major_locator(plt.NullLocator())
    i = 0
    while i < len(low_dim_embs):

        if i in id_to_word:
            
            x, y = low_dim_embs[i, :]
            word = id_to_word[i]

            if word in top_words:
                plt.scatter(x, y, color='b')
                plt.annotate(word,
                            xy=(x, y),
                            xytext=(5, 2),
                            textcoords='offset points',
                            ha='right',
                            va='bottom',
                            fontsize=14)
        
        i += 1

plot_embeddings(pca, id_to_word, top_words, figsize=(18, 18))
pyplot.show()
plot_embeddings(pca, id_to_word, get_top_words(word_t, 20))
pyplot.show()