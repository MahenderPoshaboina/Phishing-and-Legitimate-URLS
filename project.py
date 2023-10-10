import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import plot_tree

import matplotlib.pyplot as plt
import re
df = pd.read_csv('new_data_urls.csv')
df.head()
print(len(df[df['status']==0]), len(df[df['status']==1]))
df_maj, df_min = df[df['status']==1], df[df['status']==0]
df_maj_sampled = df_maj.sample(len(df_min), random_state=42)
df_balanced = pd.concat([df_maj_sampled, df_min])
print(len(df_balanced[df_balanced['status']==0]), len(df_balanced[df_balanced['status']==1]))
df_balanced.reset_index(inplace=True, drop=True)
print(df_balanced)


def tok(string) -> str:
    return string.replace('/', '.').split('.')

def to_txt(text) -> str:
    return text.replace('.', ' ').replace('/', ' ')


def top_terms(df, n) -> list[str]:
    term = {}
    for url, status in df.values:
        for word in tok(url):
            if word != '':
                if word not in term.keys():
                    term[word] = 0

                term[word]+=1
            
    return [t[0] for t in sorted(term.items(), key=lambda x:x[1], reverse=True)[:n]]
def num_digits(text) -> int:
    return len(re.findall('\d', text))

def num_dots(text) -> int:
    return len(re.findall('\.', text))

def num_bar(text) -> int:
    return len(re.findall('/', text))

VOC = top_terms(df_balanced, n=10)
print(VOC)


CORPUS = [to_txt(url) for url in df_balanced.url]
print(CORPUS[101])

vectorizer = CountVectorizer(binary=True, vocabulary=VOC)
docTermMatrix = vectorizer.fit_transform(CORPUS)

matrix = pd.DataFrame(docTermMatrix.A, columns=VOC)
matrix['dots'] = [num_dots(text) for text in df_balanced.url]
matrix['bar'] = [num_bar(text) for text in df_balanced.url]
matrix['len'] = [len(text) for text in CORPUS]
matrix['digits'] = [num_digits(text) for text in CORPUS]
print(matrix)

X_train, X_test, y_train, y_test = train_test_split(matrix.values, df_balanced['status'].values, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


confusion = confusion_matrix(y_test, y_pred)
print(confusion)

def is_legit(url):

    corpus = [to_txt(url)]
    vectorizer = CountVectorizer(binary=True, vocabulary=VOC)
    docTermMatrix = vectorizer.fit_transform(corpus)

    matrix = pd.DataFrame(docTermMatrix.A, columns=VOC)
    matrix['dots'] = [num_dots(url)]
    matrix['bar'] = [num_bar(url)]
    matrix['len'] = [len(corpus[0])]
    matrix['digits'] = [num_digits(corpus[0])]

    prediction = clf.predict(matrix.values)

    return prediction[0] == 0

print(is_legit('https://google.com/'))

plot_tree(clf, max_depth=2, feature_names=matrix.columns, class_names=['phishing', 'clean'], 
          fontsize=7)
plt.figure(figsize=(30, 30))
plt.show()

