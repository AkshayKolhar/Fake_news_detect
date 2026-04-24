

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.svm import LinearSVC


fake=pd.read_csv("fake.csv")
true=pd.read_csv("true.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

data=pd.concat([fake,true],axis=0)

data= data.sample(frac=1, random_state=42).reset_index(drop=True)


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


if result[0] == 0:
    real_news = data[data["fake"] == 1]["text"]

    real_vectors = vectorizer.transform(real_news)
    input_vector = vectorizer.transform([text])

    # cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(input_vector, real_vectors)

    most_similar_index = similarities.argmax()
    print("\n Similar REAL news example:")
    real=real_news.iloc[most_similar_index]
    print(real[:400]+"...")
    