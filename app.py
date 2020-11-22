from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/index/', methods=['post', 'get'])
def login():
    message = ''
    if request.method == 'POST':
        post = request.form.get('text')  # access the data inside 

        if len(post) <= 100:
            message = "Please tell us more about yourself"

import pandas as pd            
import numpy as np
import re
import string

# outfilename = 'mbti_preprocessed.csv'
newrows = []

def filter_text(post):
    """Decide whether or not we want to use the post."""
    # should remove link only posts here
    return len(post) > 0
    
reg_punc = re.compile('[%s]' % re.escape(string.punctuation))
def preprocess_text(post):
    """Remove any junk we don't want to use in the post."""
    
    # Remove links
    post = re.sub(r'http\S+', '', post, flags=re.MULTILINE)
    
    # All lowercase
    post  = post.lower()
    
    # Remove puncutation
    post = reg_punc.sub('', post)

    return post

def create_new_rows(row):
    posts = row['posts']
    rows = []
    
    # for p in posts:
    p = preprocess_text(posts)
    rows.append({'post': p})
    return rows

    for index, row in pdf.iterrows():
        newrows += create_new_rows(row)
        
        pdf = pd.DataFrame(newrows)

        df = spark.createDataFrame(pdf)


    from pyspark.sql.functions import length
    # Create a length column to be used as a future feature 
    df = df.withColumn('length', length(df['post']))

    types = ['INFJ', 'ENTP', 'INTP', 'INTJ', 'ENTJ', 'ENFJ', 'INFP', 'ENFP', 'ISFP', 'ISTP', 'ISFJ', 'ISTJ', 'ESTP', 'ESFP', 'ESTJ', 'ESFJ']
    types = [x.lower() for x in types]

    from pyspark.ml.feature import Tokenizer
    tokenizer = Tokenizer(inputCol="post", outputCol="words")
    tokenized = tokenizer.transform(df)

    from pyspark.ml.feature import StopWordsRemover
    # Remove stop words
    stopwordList = types
    stopwordList.extend(StopWordsRemover().getStopWords())
    stopwordList = list(set(stopwordList))#optionnal
    remover=StopWordsRemover(inputCol="words", outputCol="filtered" ,stopWords=stopwordList)
    newFrame = remover.transform(tokenized)

    from pyspark.ml.feature import HashingTF,IDF
    # Run the hashing term frequency
    hashing = HashingTF(inputCol="filtered", outputCol="hashedValues")
    # Transform into a DF 
    hashed_df = hashing.transform(newFrame) 

    # Fit the IDF on the data set 
    idf = IDF(inputCol="hashedValues", outputCol="idf_token")
    idfModel = idf.fit(hashed_df)
    rescaledData = idfModel.transform(hashed_df)

    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.linalg import Vector
    # Create feature vectors
    #idf = IDF(inputCol='hash_token', outputCol='idf_token')
    clean_up = VectorAssembler(inputCols=['idf_token', 'length'], outputCol='features')
    output = clean_up.transform(rescaledData)


    from tensorflow.keras.models import load_model

    ei_model = load_model("static/models/EI_predictor.h5")
    sn_model = load_model("static/models/SN_predictor.h5")
    tf_model = load_model("static/models/TF_predictor.h5")
    jp_model = load_model("static/models/JP_predictor.h5")






    return render_template('index.html', message=message)


if __name__ == '__main__':
    app.run(debug=True)