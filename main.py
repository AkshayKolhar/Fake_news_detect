from unittest import result

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import LinearSVC


data=pd.read_csv("fake_or_real_news.csv")


data["fake"]=data["label"].apply(lambda x:1 if x=="REAL" else 0)
data=data.drop("label", axis=1)

x,y=data["text"], data["fake"]

x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.2)


vectorizer=TfidfVectorizer(stop_words="english", max_df=0.7)
x_train_vectorized=vectorizer.fit_transform(x_train)
x_test_vectorized=vectorizer.transform(x_test)

clf=LinearSVC()
clf.fit(x_train_vectorized, y_train)


text=input("Enter the news you want to check: ")
    

vectorizer_input=vectorizer.transform([text])

result=clf.predict(vectorizer_input)

if result[0]==1:
    print("The news is real")
else:    print("The news is fake")
