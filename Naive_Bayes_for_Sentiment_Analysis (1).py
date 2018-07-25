
# coding: utf-8

# # Naive Bayes for Sentiment Analysis

# We need two external packages: `nltk` and `scikit-learn` (=`sklearn`)
#
#
# You can download and install them with `pip install nltk scikit-learn`
#
# (or `pip install --user nltk scikit-learn`)
#
# _Note: We are running this in Python 3_

# In[1]:


# NLTK
import nltk

# Machine Learning Library (http://scikit-learn.org/stable/)
import sklearn
from random import shuffle


# In[2]:


## This code downloads the required packages.
## You can run `nltk.download('all')` to download everything.

nltk_packages = [
    ("movie_reviews", "corpora/movie_reviews.zip"),
    ("punkt", "tokenizers/punkt")
]

for pid, fid in nltk_packages:
    try:
        nltk.data.find(fid)
    except LookupError:
        nltk.download(pid)


# ## Setting up corpus

# In[3]:


# Import the corpus of movie_reviews
from nltk.corpus import movie_reviews


# ## Setting up data

# In[4]:


# Load dataset into positive and negative fileids.
positive_fileids = [f for f in movie_reviews.fileids() if f.startswith("pos")]
negative_fileids = [f for f in movie_reviews.fileids() if f.startswith("neg")]

print("Number of positive reviews: {}".format(len(positive_fileids)))
print("Number of negative reviews: {}".format(len(negative_fileids)))


# In[5]:


# Create a merged list of tuples (fileid, label)
data = [(fid, 'pos') for fid in positive_fileids] + [(fid, 'neg') for fid in negative_fileids]
print(data)


# In[6]:


# Randomly shuffle the list
from random import shuffle
shuffle(data)


# In[7]:


# the data is shuffled
print(data[:5])


# ## Split into Training and Testing set

# In[8]:


# The first 80% of the data is used for training, the rest is used for testing
train_set = data[:int(0.8 * len(data))]
test_set = data[int(0.8 * len(data)):]


# In[9]:


print("Size of train set: {}".format(len(train_set)))
print("Size of test set: {}".format(len(test_set)))


# # Prepare data

# ## read data

# In[10]:


# Read data

train_documents = [movie_reviews.raw(fid) for fid, _ in train_set]
test_documents = [movie_reviews.raw(fid) for fid, _ in test_set]


# ## preprocess data

# In[11]:


# Tokenize each document

train_tokens = [nltk.word_tokenize(text) for text in train_documents]
test_tokens = [nltk.word_tokenize(text) for text in test_documents]


# In[12]:


# Get Frequency Distribution of all train tokens

freq = nltk.FreqDist(token for tokens in train_tokens for token in tokens)


# In[13]:


print(freq.most_common(10))


# ### remove stop words

# In[14]:


stop = nltk.corpus.stopwords.words('english')
print(stop)


# In[15]:


print(stop[:15])


# In[16]:


# Only use words that are not in the stopwords list.
train_tokens_no_stop = [[token for token in tokens if token.lower() not in stop] for tokens in train_tokens]


# In[17]:


# Get Frequency Distribution of train tokens without stopwords

freq_no_stop = nltk.FreqDist(token for tokens in train_tokens_no_stop for token in tokens)


# In[18]:


print(freq_no_stop.most_common(10))


# ### remove punctuation

# In[19]:


from string import punctuation


# In[20]:


# Only use words that are not punctuation
train_tokens_no_stop_punct = [[token for token in tokens if any(t not in punctuation for t in token)] for tokens in train_tokens_no_stop]


# In[21]:


# Get Frequency Distribution of train tokens without stopwords and punctuation

freq_no_stop_punct = nltk.FreqDist(token for tokens in train_tokens_no_stop_punct for token in tokens)


# In[22]:


print(freq_no_stop_punct.most_common(20))


# # Feature Extraction

# In[38]:


# Select 100 Most Common Words/Tokens as features
# Note, that we are ONLY using the frequency of the words in the TRAIN set.

word_features = [w for w, _ in freq_no_stop_punct.most_common(1000)]


# ### Feature Extraction

# In[39]:


# Create training and testing set

## Each movie review will be transformed to a feature vector of FALSE and TRUE,
## depending if the document contains the selected feature word or not.

X_train = [[w in tokens for w in word_features] for tokens in train_tokens]
X_test  = [[w in tokens for w in word_features] for tokens in test_tokens]


# In[40]:


# labels:
y_train = [label for _, label in train_set]
y_test = [label for _, label in test_set]


# In[41]:


print("First document: \n'{:.100}...'".format(train_documents[0]))


# In[27]:


print("Feature vector: \n{}\n".format(X_train[0]))


# In[42]:


print("Actual label: {}".format(y_train[0]))


# # Classification

# ### Classifier Training

# In[43]:


# From the scikit-learn library, we are using Bernoulli Naive Bayes

from sklearn.naive_bayes import GaussianNB


# In[44]:


#

clf = GaussianNB()


# In[45]:


# Train (= Fit) classifier with training data

clf.fit(X=X_train, y=y_train)


# ### Classifier Evaluation

# In[46]:


# Run prediction on test set

y_pred = clf.predict(X_test)


# Now `y_pred` contains the predicted label for each of the movie reviews from the test set.

# In[48]:


# For example, the first movie review
print("Predicted: {}".format(y_pred[120]))
print("Actual   : {}".format(y_test[120]))


# In[49]:


# We can evaluate the performance with functionality from the scikit-learn library.
from sklearn import metrics


# In[50]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print (accuracy)


# In[49]:


print("Accuracy : {:.4f}".format(metrics.accuracy_score(y_test, y_pred)))
print("Precision: {:.4f}".format(metrics.precision_score(y_test, y_pred, pos_label='pos')))
print("Recall   : {:.4f}".format(metrics.recall_score(y_test, y_pred, pos_label='pos')))
print("F1-Score : {:.4f}".format(metrics.f1_score(y_test, y_pred, pos_label='pos')))


# In[50]:


print(metrics.classification_report(y_true=y_test, y_pred=y_pred, digits=3))


# ## Predict from any text:

# In[51]:


def do_prediction_on_text(text):

    # Tokenize
    example_tokens = nltk.word_tokenize(text)

    # Extract features
    example_features = [w in example_tokens for w in word_features]

    # Do prediction
    example_pred = clf.predict([example_features])[0]

    return example_pred


# In[52]:


# Example sentences:
example_text = "I hated this movie so much that I left the cinema after five minutes of this garbage."

example_pred = do_prediction_on_text(example_text)

print("Example text: '{}'".format(example_text))
print("Predicted sentiment: '{}'\n".format(example_pred))


# # Possible Improvements
#
# - Use not the 100 most common words, but something in the middle.
# - Use the word count instead of Binary values
# - Use a different classifier (Logistic Regression, SVM)
# - Identify key words and give them higher weight

# ## How stopword removal impacts

# In[53]:


# Without stopword removal, without punctuation removal:
X_train = [[w in tokens for w, _ in freq.most_common(100)] for tokens in train_tokens]
X_test  = [[w in tokens for w, _ in freq.most_common(100)] for tokens in test_tokens]

clf = BernoulliNB()
clf.fit(X_train, y_train)
print("Accuracy without any word removal: {:.4f}".format(clf.score(X_test, y_test)))


# In[56]:


# With stopword removal, without punctuation removal:
X_train = [[w in tokens for w, _ in freq_no_stop.most_common(100)] for tokens in train_tokens]
X_test  = [[w in tokens for w, _ in freq_no_stop.most_common(100)] for tokens in test_tokens]

clf = BernoulliNB()
clf.fit(X_train, y_train)
print("Accuracy with stopword removal: {:.4f}".format(clf.score(X_test, y_test)))


# In[57]:


# With stopword removal, with punctuation removal:
X_train = [[w in tokens for w, _ in freq_no_stop_punct.most_common(100)] for tokens in train_tokens]
X_test  = [[w in tokens for w, _ in freq_no_stop_punct.most_common(100)] for tokens in test_tokens]

clf = BernoulliNB()
clf.fit(X_train, y_train)
print("Accuracy with stopword/punctuation removal: {:.4f}".format(clf.score(X_test, y_test)))

