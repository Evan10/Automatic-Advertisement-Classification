############################################ Import Libraries###########################################################

import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Conv1D, MaxPooling1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
import re
from nltk.corpus import stopwords
import numpy as np

########################################################################################################################

############################################ Data Preprocessing ########################################################

def data_prep(car,general):

    colnames = ['Text', 'Label']
    df_gen = pd.read_csv(general, usecols=colnames)  # for general data
    df_car = pd.read_csv(car,usecols=colnames)  # for car data
    df_gen.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    df_car.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    reviews_gen = df_gen['Text'].tolist()
    reviews_car = df_car['Text'].tolist()
    lemmatizer = nltk.stem.WordNetLemmatizer()

    def pre_process_text(text):
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens if token.isalpha()]
        stop_words_removed = [token for token in lemmatized if not token in stopwords.words('english')]
        stop = " ".join(stop_words_removed)
        return stop

    reviews_gen = list(map(pre_process_text, reviews_gen))
    reviews_car = list(map(pre_process_text, reviews_car))
    n = 'Reviews_Proc'
    n1 = df_gen.columns[0]
    df_gen.drop(n1, axis=1, inplace=True)
    df_gen[n] = reviews_gen
    n2 = df_car.columns[0]
    df_car.drop(n2, axis=1, inplace=True)
    df_car[n] = reviews_car
    df_car['Coded'] = pd.get_dummies(df_car['Label'], drop_first=True)
    df_gen['Coded'] = pd.get_dummies(df_gen['Label'], drop_first=True)
    return df_car,df_gen

# training block
car = 'C:\\Users\\tirth\\OneDrive\\Desktop\\Coursework\\AUD\\Project\\cars.csv'
gen = 'C:\\Users\\tirth\\OneDrive\\Desktop\\Coursework\\AUD\\Project\\general.csv'
df_car, df_gen = data_prep(car,gen)
data_gen, data_car = df_gen.values.tolist(), df_car.values.tolist()
train_gen, test_gen = data_gen[:1500] , data_gen[1500:]  # split train and test set for restaurants
train_car, test_car = data_car[:1500], data_car[1500:]  # split train and test set for movies
training, testing = train_gen + train_car, test_gen + test_car  # combine training and test sets
x_train, x_test = [i[1] for i in training], [i[1] for i in testing]
y_train, y_test = [i[2] for i in training], [i[2] for i in testing]
car = 'C:\\Users\\tirth\\OneDrive\\Desktop\\Coursework\\AUD\\Project\\tippe_cars.csv'
gen = 'C:\\Users\\tirth\\OneDrive\\Desktop\\Coursework\\AUD\\Project\\tippe_general.csv'
df_car, df_gen = data_prep(car,gen)
df = pd.concat([df_gen,df_car])
df_x = df['Reviews_Proc'].tolist()
df_y = df['Coded'].tolist()

########################################################################################################################

############################################ Base Vectorizer & DL Model ################################################

def TFIDF(ran,min, train, test):

    def tk(doc):
        return doc
    ran = (1,ran)
    print(ran)
    tfidf = TfidfVectorizer(analyzer='word', preprocessor=tk, tokenizer=tk, ngram_range=ran,
                            min_df=min) # initialize TF-IDF vectorizer
    tfidf.fit(train)
    x_train = tfidf.transform(train)
    x_test = tfidf.transform(test)
    print("For min_df=",min,"and ngram=",ran)
    return x_train,x_test


def deep(i,j,x_train,y_train,x_test,y_test):

    DLmodel2 = MLPClassifier(solver='adam', hidden_layer_sizes=(i,j), random_state=1, n_iter_no_change=20,
                             learning_rate='adaptive', max_iter=200, verbose=False, early_stopping=True)
    DLmodel2.fit(x_train, y_train)
    y_pred_DL = DLmodel2.predict(x_test)
    acc_DL = accuracy_score(y_test, y_pred_DL)

    print("For nodes = ",i,"and layers = ", j," accuracy=  ",acc_DL)
    return acc_DL

########################################################################################################################

############################################ Hyperparameter Tuning #####################################################

best=0
best_min_df=0
best_ngram = 0
best_nodes = 0
best_layers = 0

for min_df in [3,4,5]:
    for n_gram in [2,3,4]:
        train,test = TFIDF(n_gram,min_df,x_train,x_test)
        for nodes in list(np.arange(5,40,5)):
            for layers in [2,3,4,5,6]:
                acc = deep(nodes,layers,train,y_train,test,y_test)
                if acc>best:
                    best=acc
                    best_min_df=min_df
                    best_ngram = n_gram
                    best_nodes = nodes
                    best_layers = layers

print("Best Paramters are :")
print("Min_df :",best_min_df)
print("N_gram :",best_ngram)
print("Nodes :",best_nodes)
print("Layers :",best_layers)
print("Accuracy :", best)

########################################################################################################################

############################################ Optimized Vectorizer ######################################################

def tk(doc):
    return doc

tfidf = TfidfVectorizer(analyzer='word', preprocessor=tk, tokenizer=tk, ngram_range=(1,4),
                            min_df=3)

tfidf.fit(x_train)
x_train = tfidf.transform(x_train)
x_test = tfidf.transform(x_test)
x_tip = tfidf.transform(df_x)

########################################################################################################################

############################################ Supervised Learning #######################################################
DLmodel2 = MLPClassifier(solver='adam', hidden_layer_sizes=(15,3), random_state=1, n_iter_no_change=20,
                        learning_rate='adaptive', max_iter=200, verbose=False, early_stopping=True)
DLmodel2.fit(x_train, y_train)
y_pred_DL= DLmodel2.predict(x_test)
y_pred_tip = DLmodel2.predict(x_tip)
y_pred_tr = DLmodel2.predict(x_train)
acc_DL = accuracy_score(y_test, y_pred_DL)
acc_DL_tip = accuracy_score(df_y, y_pred_tip)
acc_DL_tr = accuracy_score(y_train, y_pred_tr)
print("DL model Train Accuracy: {:.2f}%".format(acc_DL_tr*100))
print("DL model Validation Accuracy: {:.2f}%".format(acc_DL*100))
print("DL model Test Accuracy: {:.2f}%".format(acc_DL_tip*100))


# Logit
Logitmodel = LogisticRegression()
Logitmodel.fit(x_train, y_train)
y_pred_logit_tr = Logitmodel.predict(x_train)
y_pred_logit_vl = Logitmodel.predict(x_test)
y_pred_logit_te = Logitmodel.predict(x_tip)
acc_logit_tr = accuracy_score(y_train, y_pred_logit_tr)
acc_logit_vl = accuracy_score(y_test, y_pred_logit_vl)
acc_logit_te = accuracy_score(df_y, y_pred_logit_te)

print("Logit model Train Accuracy:: {:.2f}%".format(acc_logit_tr*100))
print("Logit model Validation Accuracy:: {:.2f}%".format(acc_logit_vl*100))
print("Logit model Test Accuracy:: {:.2f}%".format(acc_logit_te*100))


# Naive Bayes
NBmodel = MultinomialNB()
NBmodel.fit(x_train, y_train)
y_pred_nb_tr = NBmodel.predict(x_train)
y_pred_nb_vl = NBmodel.predict(x_test)
y_pred_nb_te = NBmodel.predict(x_tip)
acc_nb_tr = accuracy_score(y_train, y_pred_nb_tr)
acc_nb_vl = accuracy_score(y_test, y_pred_nb_vl)
acc_nb_te = accuracy_score(df_y, y_pred_nb_te)
print("NB model Train Accuracy:: {:.2f}%".format(acc_nb_tr*100))
print("NB model Validation Accuracy:: {:.2f}%".format(acc_nb_vl*100))
print("NB model Test Accuracy:: {:.2f}%".format(acc_nb_te*100))



# Support Vector Classifier
SVMmodel = LinearSVC()
SVMmodel.fit(x_train, y_train)
y_pred_svc_tr = SVMmodel.predict(x_train)
y_pred_svc_vl = SVMmodel.predict(x_test)
y_pred_svc_te = SVMmodel.predict(x_tip)
acc_svc_tr = accuracy_score(y_train, y_pred_svc_tr)
acc_svc_vl = accuracy_score(y_test, y_pred_svc_vl)
acc_svc_te = accuracy_score(df_y, y_pred_svc_te)
print("SVC model Train Accuracy:: {:.2f}%".format(acc_svc_tr*100))
print("SVC model Validation Accuracy:: {:.2f}%".format(acc_svc_vl*100))
print("SVC model Test Accuracy:: {:.2f}%".format(acc_svc_te*100))



# Random Forest Classifier
RFmodel = RandomForestClassifier(n_estimators=100, max_depth=10, criterion='entropy',
                                 bootstrap=True, random_state=0) ## number of trees and number of layers/depth
RFmodel.fit(x_train, y_train)
y_pred_rf_tr = RFmodel.predict(x_train)
y_pred_rf_vl = RFmodel.predict(x_test)
y_pred_rf_te = RFmodel.predict(x_tip)
acc_rf_tr = accuracy_score(y_train, y_pred_rf_tr)
acc_rf_vl = accuracy_score(y_test, y_pred_rf_vl)
acc_rf_te = accuracy_score(df_y, y_pred_rf_te)
print("SVC model Train Accuracy:: {:.2f}%".format(acc_rf_tr*100))
print("SVC model Validation Accuracy:: {:.2f}%".format(acc_rf_vl*100))
print("SVC model Test Accuracy:: {:.2f}%".format(acc_rf_te*100))

########################################################################################################################

################################################### LSTM ###############################################################

def clean_text(text):
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text)  # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing.
    text = text.replace('x', '')
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # remove stopwors from text
    return text

STOPWORDS = set(stopwords.words('english'))
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
MAX_NB_WORDS = 50000 # The maximum number of words to be used. (most frequent)
MAX_SEQUENCE_LENGTH = 250 # Max number of words in each complaint.
EMBEDDING_DIM = 100 # This is fixed.
colnames = ['Text', 'Label']
car = 'C:\\Users\\tirth\\OneDrive\\Desktop\\Coursework\\AUD\\Project\\cars.csv'
general = 'C:\\Users\\tirth\\OneDrive\\Desktop\\Coursework\\AUD\\Project\\general.csv'
df_gen = pd.read_csv(general, usecols=colnames)  # for general data
df_car = pd.read_csv(car,usecols=colnames)  # for car data
df = pd.concat([df_gen, df_car])
df['Text'] = df['Text'].astype(str)
df['Text'] = df['Text'].apply(clean_text)
df['Text'] = df['Text'].str.replace('\d+', '')

tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['Text'].values)
X = tokenizer.texts_to_sequences(df['Text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(df['Label']).values
word_index = tokenizer.word_index
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.20, random_state = 42)

print('Found %s unique tokens.' % len(word_index))
print('Shape of data tensor:', X.shape)
print('Shape of label tensor:', Y.shape)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(140, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 2
batch_size = 60
history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,
                    callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

accr = model.evaluate(X_test,Y_test)
print('Validation set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

########################################################################################################################

############################################### LSTM Evaluation ########################################################

car = 'C:\\Users\\tirth\\OneDrive\\Desktop\\Coursework\\AUD\\Project\\tippe_cars.csv'
gen = 'C:\\Users\\tirth\\OneDrive\\Desktop\\Coursework\\AUD\\Project\\tippe_general.csv'
df = pd.concat([df_gen,df_car])
df['Text'] = df['Text'].astype(str)
df['Text'] = df['Text'].apply(clean_text)
df['Text'] = df['Text'].str.replace('\d+', '')
tokenizer.fit_on_texts(df['Text'].values)
X = tokenizer.texts_to_sequences(df['Text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
Y = pd.get_dummies(df['Label']).values
word_index = tokenizer.word_index

accr = model.evaluate(X, Y)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
########################################################################################################################