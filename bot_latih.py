import json
import telebot
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pandas as pd
import string
import numpy as np
import random
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Flatten, Dense
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Inisialisasi bot
bot_token = '5876855745:AAHgS6ZiRe30GbwVYhvqNEfvjjfQKTMotiQ'
bot = telebot.TeleBot(bot_token)

# Fungsi untuk menambahkan data baru ke file JSON intents
def tambah_data_pertanyaan(pertanyaan, tag, respons):
    # Membaca file JSON intents
    with open('intents.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Menambahkan data baru ke dalam intents
    new_data = {
        "tag": tag,
        "patterns": [pertanyaan],
        "responses": [respons]
    }
    data['intents'].append(new_data)

    # Menulis kembali file JSON dengan data baru
    with open('intents.json', 'w', encoding='utf-8') as file:
        json.dump(data, file, separators=(',', ': '), indent=4)

    # Import the dataset
    with open('intents.json', 'r', encoding='utf-8') as content:
        data = json.load(content)

    # Get all data into lists
    tags = []
    inputs = []
    responses = {}
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!']

    for intent in data['intents']:
        responses[intent['tag']] = intent['responses']
        for pattern in intent['patterns']:
            inputs.append(pattern)
            tags.append(intent['tag'])
            words.extend(nltk.word_tokenize(pattern))
            documents.append((nltk.word_tokenize(pattern), intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    # Convert data to a dataframe
    data = pd.DataFrame({"patterns": inputs, "tags": tags})
    data['patterns'] = data['patterns'].apply(lambda wrd: [ltrs.lower() for ltrs in nltk.word_tokenize(wrd) if ltrs not in string.punctuation])
    data['patterns'] = data['patterns'].apply(lambda wrd: ' '.join(wrd))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    print(len(words), "unique lemmatized words:", words)
    classes = sorted(list(set(classes)))
    print(len(classes), "classes:", classes)
    print(len(documents), "documents")

    # Tokenize the data
    tokenizer = Tokenizer(num_words=2000)
    tokenizer.fit_on_texts(data['patterns'])
    train = tokenizer.texts_to_sequences(data['patterns'])
    x_train = pad_sequences(train)
    print(x_train)

    # Encode the outputs
    le = LabelEncoder()
    y_train = le.fit_transform(data['tags'])
    print(y_train)

    input_shape = x_train.shape[1]
    print("Input shape:", input_shape)

    vocabulary = len(tokenizer.word_index)
    print("Number of unique words:", vocabulary)

    output_length = le.classes_.shape[0]
    print("Output length:", output_length)

    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))
    pickle.dump(le, open('le.pkl', 'wb'))
    pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))

    training = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)
        
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array([item[0] for item in training])
    train_y = np.array([item[1] for item in training])

    i = Input(shape=(input_shape,))
    x = Embedding(vocabulary + 1, 10)(i)
    x = LSTM(10, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(output_length, activation="softmax")(x)
    model = Model(i, x)

    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    model.summary()

    train = model.fit(x_train, y_train, epochs=400)

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train.history['accuracy'], label='Training Set Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(train.history['loss'], label='Training Set Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    plt.savefig('training_plot.png')

    # Save the models
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))
    pickle.dump(le, open('le.pkl', 'wb'))
    pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
    model.save('RPTRABot_model.h5')

    # Load the models
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    le = pickle.load(open('le.pkl', 'rb'))
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

    # Load the trained model
    model = tf.keras.models.load_model('RPTRABot_model.h5')

    return "Data telah ditambahkan!"

# Fungsi handler untuk perintah tambah data
@bot.message_handler(commands=['tambah_data'])
def tambah_data_handler(message):
    # Meminta input pertanyaan, tag, dan respons dari pengguna
    bot.reply_to(message, "Masukkan pertanyaan baru:")
    bot.register_next_step_handler(message, tambah_data_pertanyaan_1)

def tambah_data_pertanyaan_1(message):
    pertanyaan = message.text
    bot.reply_to(message, "Masukkan tag untuk pertanyaan tersebut:")
    bot.register_next_step_handler(message, tambah_data_pertanyaan_2, pertanyaan)

def tambah_data_pertanyaan_2(message, pertanyaan):
    tag = message.text
    bot.reply_to(message, "Masukkan respons untuk pertanyaan tersebut:")
    bot.register_next_step_handler(message, tambah_data_pertanyaan_3, pertanyaan, tag)

def tambah_data_pertanyaan_3(message, pertanyaan, tag):
    respons = message.text

    # Menambahkan data baru ke file JSON intents
    result = tambah_data_pertanyaan(pertanyaan, tag, respons)
    bot.reply_to(message, result)
    bot.send_photo(message.chat.id, open('training_plot.png', 'rb'))

# Menjalankan bot
bot.polling()
