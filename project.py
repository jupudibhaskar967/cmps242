import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

N=100
print N
with open('yelp_academic_dataset_review.json', 'r') as f:
	data = f.readlines()[0:N]
	
data = map(lambda x: x.rstrip(), data)
data_json_str = "[" + ','.join(data) + "]"
print data_json_str[0:100]
df = pd.read_json(data_json_str)
#print df
df[['stars','text']].head(10)
#print df

train_labels = []
train_data = df['text'].head(N/2).tolist()
for n in df['stars'].head(N/2).as_matrix():
	res = 'pos' if n > 3 else 'neg'
	train_labels.append(res)

# Generate testing data
test_labels = []
test_data = df['text'].tail(N/2).tolist()
for n in df['stars'].tail(N/2).as_matrix():
	res = 'pos' if n > 3 else 'neg'
	test_labels.append(res)
	


count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(train_data)
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
clf = MultinomialNB().fit(train_tfidf, train_labels)


test_counts = count_vect.transform(test_data)
test_tfidf = tfidf_transformer.transform(test_counts)
predicted = clf.predict(test_tfidf)
print('%.1f%%' % (np.mean(predicted == test_labels) * 100))





	
	
