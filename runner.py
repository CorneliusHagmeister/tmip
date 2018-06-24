import pandas as pd
from gensim.models import KeyedVectors
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM
from keras.utils import to_categorical
from nltk.tokenize import RegexpTokenizer
from keras.models import load_model
import logging
import os
import tensorflow as tf
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True
from keras.backend.tensorflow_backend import set_session

import config


def load_comments(filename, num_chunks=None):
    print("loading comments...")
    chunksize = 10 ** 6
    comments_list = []
    i = 0
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        comment_chunk = chunk[chunk['comment_text'].notnull()]  # filter out NaNs
        if num_chunks is None or i < num_chunks:
            comment_chunk = comment_chunk[comment_chunk['parent_comment_id'].isnull()]  # only select root comments
            comment_chunk = comment_chunk.drop(['parent_comment_id'], axis=1)
            print(comment_chunk.shape)
            comments_list.append(comment_chunk)
            print(len(comments_list))
            i += 1
        else:
            break
    comments = pd.concat(comments_list)
    comments['cluster'] = comments["upvotes"].apply(get_class)
    print(
    "clusters: ", "0: ", len(comments[comments["cluster"] == 0]), "1: ", len(comments[comments["cluster"] == 1]), "2: ",
    len(comments[comments["cluster"] == 2]))
    return comments


def load_constructiveness_comments(filename, num_chunks=None):
    print("loading comments...")
    chunksize = 10 ** 6
    comments_list = []
    i = 0
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        comment_chunk = chunk[chunk['comment_text'].notnull()]  # filter out NaNs
        if num_chunks is None or i < num_chunks:
            comment_chunk = comment_chunk[comment_chunk['comment_text'].notnull()]  # only select root comments
            # comment_chunk = comment_chunk.drop(['comment_text'], axis=1)
            print(comment_chunk.shape)
            comments_list.append(comment_chunk)
            print(len(comments_list))
            i += 1
        else:
            break
    comments = pd.concat(comments_list)
    comments['cluster'] = comments["is_constructive"].apply(is_constructive)
    print(
        "clusters: ", "0: ", len(comments[comments["cluster"] == 0]), "1: ", len(comments[comments["cluster"] == 1]))
    return comments

def load_gnm_comments(filename,num_chunks):
    print("loading comments...")
    chunksize = 10 ** 6
    comments_list = []
    i = 0
    for chunk in pd.read_csv(filename, chunksize=chunksize):
        comment_chunk = chunk[chunk['comment_text'].notnull()]  # filter out NaNs
        if num_chunks is None or i < num_chunks:
            comment_chunk = comment_chunk[comment_chunk['comment_text'].notnull()]  # only select root comments
            # comment_chunk = comment_chunk.drop(['comment_text'], axis=1)
            print(comment_chunk.shape)
            comments_list.append(comment_chunk)
            print(len(comments_list))
            i += 1
        else:
            break
    comments = pd.concat(comments_list)
    return comments

def is_constructive(is_constructive):
    if is_constructive == "yes":
        return 1
    else:
        return 0


def get_class(upvotes):
    if upvotes > 0 and upvotes < 4:
        return 0
    elif upvotes < 5:
        return 1
    else:
        return 2


def load_embedding():
    print("loading embeddings...")
    # word_vectors = KeyedVectors.load_word2vec_format('wordvectors.w2v')
    # word_vectors.save("wordvectors.bin")
    # word_vectors.save_word2vec_format('wordvectors.w2v')
    word_vectors = KeyedVectors.load("wordvectors.bin")
    return word_vectors


def preprocess_data(docs):
    tokenizer = RegexpTokenizer(r'\w+')
    MAX_NB_WORDS = 100000
    max_seq_len = 40
    print("pre-processing train data...")
    processed_docs = []
    print("tokenizing input data...")
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
    tokenizer.fit_on_texts(processed_docs)  # leaky
    word_seq = tokenizer.texts_to_sequences(processed_docs)
    word_index = tokenizer.word_index
    print("dictionary size: ", len(word_index))

    # pad sequences
    word_seq = sequence.pad_sequences(word_seq, maxlen=max_seq_len)
    return word_seq, word_index


def custom_tokenize(docs, word_vectors):
    # should probably also be done once and saved
    print("tokenizing docs...")
    output_matrix = []
    errors = 0
    for d in docs:
        indices = []
        for w in d:
            try:
                indices.append(word_vectors.vocab[w].index)
            except KeyError:
                errors += 1
        output_matrix.append(indices)
    print(errors, " words not found in vocab")
    return output_matrix


def back2text(doc, word_vectors):
    sentence = ""
    for word_index in doc:
        sentence = sentence + word_vectors.index2word[word_index] + " "
    return sentence


def define_train_model(config, embedding_layer):
    # CNN architecture
    print("training CNN ...")
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(config.num_filters, 7, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(config.num_filters, 7, activation='relu', padding='same'))
    model.add(GlobalMaxPooling1D())
    # model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(30, activation='relu', kernel_regularizer=regularizers.l2(config.weight_decay)))
    model.add(Dense(config.num_classes, activation='sigmoid'))  # multi-label (k-hot encoding)

    adam = optimizers.Adam(lr=config.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.summary()

    return model


def main():
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    tfconfig.log_device_placement = True  # to log device placement (on which device the operation ran)
    # (nothing gets printed in Jupyter, only if you run it standalone)
    sess = tf.Session(config=tfconfig)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    # nltk.download('stopwords')
    app_config = config.Config

    print("loading data...")
    # comments = load_comments('../sorted_comments-standardized-tokenized/sorted_comments-standardized-tokenized.csv', 5)
    # comments = load_constructiveness_comments(
    #     "../SOCC/annotated/constructiveness/SFU_constructiveness_toxicity_corpus.csv", 5)
    comments = load_gnm_comments("../SOCC/raw/gnm_comments.csv", 5)
    diff_metrics_data=utils.create_training_data(comments)
    print("loading done")
    docs = comments['comment_text'].tolist()
    docs_processed = []
    for entry in docs:
        sentence = []
        for word in entry.split():
            sentence.append(word)
        docs_processed.append(sentence)

    word_vectors = load_embedding()
    print("processing data...")

    encoded_docs = custom_tokenize(docs_processed, word_vectors)
    padded_docs_ge = pad_sequences(encoded_docs, maxlen=app_config.padding_len, padding='post')
    padded_docs_ge_train=padded_docs_ge[:800]
    target_values_train= comments['cluster'].tolist()[:800]

    model = None
    if app_config.load_model is True:
        print("loading model...")
        model = load_model(app_config.load_model_path)
    else:

        print("creating model...")

        embedding_layer = word_vectors.get_keras_embedding(train_embeddings=False)
        model = define_train_model(app_config, embedding_layer)

        # define callbacks

        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
        # callbacks_list = [early_stopping]
        callbacks_list = []

        # model training
        hist = model.fit(padded_docs_ge_train, to_categorical(target_values_train), batch_size=app_config.batch_size,
                         epochs=app_config.num_epochs, callbacks=callbacks_list,
                         validation_split=0.3, shuffle=True, verbose=3)
        if app_config.save_model is True:
            print("saving model...")
            model.save(app_config.save_model_path)
    print("lets predict something...")
    padded_docs_ge_test =padded_docs_ge[800:]
    y_test = model.predict(padded_docs_ge_test)
    y_label = model.predict_classes(padded_docs_ge_test)
    y_label_real = comments['cluster'].tolist()
    target_values_test= comments['cluster'].tolist()[800:]

    counter=0
    for i in range(0, len(target_values_test)):
        if y_label[i]==target_values_test[i]:
            counter+=1

    print(y_test)
    print("acc: ",counter/len(target_values_test))


if __name__ == "__main__":
    main()
