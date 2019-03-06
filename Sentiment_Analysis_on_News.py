
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import gc
import sys
import itertools
import nltk
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords 
from nltk.stem.wordnet import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import rcParams

get_ipython().magic(u'matplotlib inline')


# In[2]:


sns.set(style='darkgrid', palette='dark', font_scale=2)
rcParams['figure.figsize'] = 10, 6


# In[3]:


#Read data
try:
    news_df = pd.read_csv('one.csv', encoding='unicode_escape')
except Exception as e:
    print(e)
    gc.collect()


# In[4]:


news_df.head()


# In[5]:


sns.countplot(news_df['News_Final.SentimentTitle'])
plt.xlabel('Label')
plt.title('Sentiments Count')


# ## Cleaning Text

# In[6]:


def strip_non_ascii(string):
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


# In[7]:


def clean_texts(x):
    if x:
        x = strip_non_ascii(x)
        x = BeautifulSoup(x, "lxml")
        x =  x.get_text()
        x = x.replace('\n', ' ').replace('\r', '').replace('\t',' ')
        # remove between word dashes
        x = x.replace('- ', ' ').replace(' -',' ').replace('-',' ')
        #replace parentheses
        x = x.replace("(","").replace(")","").replace("[","").replace("]","")
        #remove punctuation but keep commas, semicolons, periods, exclamation marks, question marks, intra-word dashes and apostrophes (e.g., "I'd like")
        x = x.replace(r"[^[:alnum:][:space:].'-:]", " ").replace('+',' ').replace('*',' ').replace("' ","").replace(" '","").replace("'","").replace(","," ").replace(";"," ").replace(":"," ")
        #remove numbers (integers and floats)
        x = re.sub('\d+', '', x)        
        #remove extra white space, trim and lower
        x = re.sub('\\s+',' ',x).strip()
        return x
    else:
        return ""    


# In[8]:


stopwords_set = set(stopwords.words('english'))

def remove_stop_words(doc):
    x = " ".join([word for word in doc.lower().split() if word not in stopwords_set])
    return x


# In[9]:


lemma = WordNetLemmatizer()
def lemmatize_words(doc):
    x = " ".join(lemma.lemmatize(word) for word in doc.split())
    return x


# In[10]:


def pos_tagging(text):
    good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])
    tagged_words = list(itertools.chain.from_iterable(nltk.pos_tag_sents(nltk.word_tokenize(sent)
                                                                    for sent in nltk.sent_tokenize(text))))
    # filter on certain POS tags and lower case all words
    candidates = [word.lower() for word, tag in tagged_words
                  if tag in good_tags]
    
    x = " ".join(s for s in candidates)
    return x


# In[11]:


news_df['News_Final.Title'] = news_df['News_Final.Title'].map(clean_texts)
news_df['News_Final.Title'] = news_df['News_Final.Title'].map(remove_stop_words)
news_df['News_Final.Title'] = news_df['News_Final.Title'].map(pos_tagging)
news_df['News_Final.Title'] = news_df['News_Final.Title'].map(lemmatize_words)


# In[12]:


news_df.head()


# ## Split into training and test datasets

# In[13]:


from sklearn.cross_validation import train_test_split
X_train_text, X_test_text, y_train_label, y_test_label = train_test_split(news_df['News_Final.Title'],
                                                                          news_df['News_Final.SentimentTitle'], test_size=0.2)


# ## Count Vectors

# Count Vector is a matrix notation of the dataset. It converts text documents into a matrix of token counts. It produces a sparse representation of term counts.

# In[14]:


from sklearn.feature_extraction.text import CountVectorizer
# count vectorizer object 
vector_count = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
vector_count.fit(news_df['News_Final.Title'])


# Convert training and test data using count vectorizer

# In[15]:


X_train_count =  vector_count.transform(X_train_text)
X_test_count =  vector_count.transform(X_test_text)


# ## TfIDF Vector

# Tf-Idf vector will be developed where every element represents the tf-idf score of each term. 

# #### Word Level Tf-Idf

# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer

vector_tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words='english')
vector_tfidf.fit(news_df['News_Final.Title'])


# In[17]:


X_train_tfidf =  vector_tfidf.transform(X_train_text)
X_test_tfidf =  vector_tfidf.transform(X_test_text)


# #### N-gram level Tf-Idf

# In[18]:


vector_ngram_tfidf = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,2), stop_words='english')
vector_ngram_tfidf.fit(news_df['News_Final.Title'])


# In[19]:


X_train_tfidf_ngram =  vector_ngram_tfidf.transform(X_train_text)
X_test_tfidf_ngram =  vector_ngram_tfidf.transform(X_test_text)


# ## Classification using Random Forest

# #### On word level Tf-Idf vector

# In[20]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=2000, n_jobs=-1, verbose=1, oob_score = True)
clf.fit(X_train_tfidf, y_train_label)
pred = clf.predict(X_test_tfidf)


# In[21]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test_label, pred)


# In[22]:


from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test_label, pred, labels=None, sample_weight=None)


# In[23]:


class_names = [0,1]
fontsize=14
df_conf_matrix = pd.DataFrame(
        conf_matrix, index=class_names, columns=class_names, 
    )
fig = plt.figure()
heatmap = sns.heatmap(df_conf_matrix, annot=True, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
plt.ylabel('True label')
plt.xlabel('Predicted label')


# In[24]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, threshold = roc_curve(y_test_label, pred)
roc_auc = auc(fpr, tpr)


# In[25]:


plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# #### On N-gram Tf-Idf vector

# In[26]:


clf = RandomForestClassifier(n_estimators=2000, n_jobs=-1, verbose=1, oob_score = True)
clf.fit(X_train_tfidf_ngram, y_train_label)
pred = clf.predict(X_test_tfidf_ngram)


# In[27]:


accuracy_score(y_test_label, pred)


# In[28]:


confusion_matrix(y_test_label, pred, labels=None, sample_weight=None)


# In[29]:


clf = RandomForestClassifier(n_estimators=2000, n_jobs=-1, verbose=1, oob_score = True)
clf.fit(X_train_count, y_train_label)
pred = clf.predict(X_test_count)


# In[30]:


accuracy_score(y_test_label, pred)


# In[31]:


confusion_matrix(y_test_label, pred, labels=None, sample_weight=None)


# ## Precision, Recall and F-Score measurement

# In[32]:


from sklearn.metrics import precision_recall_fscore_support

precision_recall_fscore_support(y_test_label, pred, average='binary')


# ## Plotting Precision - Recall Curve

# In[33]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test_label, pred)

precision, recall, _ = precision_recall_curve(y_test_label, pred)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


# In[34]:


from sklearn.svm import SVC

clf = SVC(gamma=1e-4,kernel='linear')
clf.fit(X_train_count, y_train_label)
pred = clf.predict(X_test_count)


# In[35]:


accuracy_score(y_test_label, pred)


# In[36]:


X_train_count.shape


# ## With decomposition

# In[37]:


from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD (n_components=450)
svd.fit(X_train_count)
X_train_count_decomp = svd.transform(X_train_count)


# In[38]:


svd.explained_variance_ratio_.sum()


# In[39]:


X_train_count_decomp.shape


# In[40]:


X_test_count.shape


# In[41]:


X_test_count_decomp = svd.transform(X_test_count)


# In[42]:


X_test_count_decomp.shape


# In[43]:


clf = SVC(gamma=1e-4,kernel='linear')
clf.fit(X_train_count_decomp, y_train_label)
pred = clf.predict(X_test_count_decomp)


# In[44]:


accuracy_score(y_test_label, pred)


# In[59]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


# In[61]:


print(precision_score(y_test_label, pred, average="macro"))
print(recall_score(y_test_label, pred, average="macro"))
print(f1_score(y_test_label, pred, average="macro"))


# ## LSTM

# In[56]:


from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D, Convolution1D, GlobalMaxPool1D
from keras.optimizers import RMSprop, Adam
from keras.preprocessing import sequence


# In[54]:


get_ipython().system(u'pip install np_utils')


# In[267]:


max_words = 1000
max_len = 450
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train_text)
sequences = tok.texts_to_sequences(X_train_text)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)


# In[298]:


def create_RNN_model():
    inputs = Input(name='inputs',shape=[max_len])
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model


# In[256]:


model = RNN()
model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])


# In[257]:


model.fit(sequences_matrix,y_train_label,batch_size=128,epochs=10,
          #validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
          validation_split=0.2)


# In[258]:


test_sequences = tok.texts_to_sequences(X_test_text)
test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)


# In[259]:


accr = model.evaluate(test_sequences_matrix,y_test_label)


# In[260]:


print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))


# In[250]:


model.summary()


# ## Convolution Neural Network

# In[304]:



def create_CNN_model():
 max_words = len(word_index) + 1

 input_layer = Input(name='inputs',shape=[max_len])
 embedding_layer = Embedding(max_words, 300, weights=[embedding_matrix], trainable=False)(input_layer)
 embedding_layer = SpatialDropout1D(0.3)(embedding_layer)

 conv_layer = Convolution1D(100, 3, activation="relu")(embedding_layer)

 pooling_layer = GlobalMaxPool1D()(conv_layer)

 output_layer1 = Dense(50, activation="relu")(pooling_layer)
 output_layer1 = Dropout(0.25)(output_layer1)
 output_layer2 = Dense(1, activation="sigmoid")(output_layer1)

 model = Model(inputs=input_layer, outputs=output_layer2)
 model.compile(optimizer=Adam(), loss='binary_crossentropy')
 
 return model


# In[305]:


cnn_model = create_cnn()


# In[306]:


cnn_model.fit(sequences_matrix,y_train_label,batch_size=128,epochs=10,validation_split=0.2)


# In[307]:


accr = cnn_model.evaluate(test_sequences_matrix,y_test_label)

print(accr)


# In[308]:


cnn_model.summary()

