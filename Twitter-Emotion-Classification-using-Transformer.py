# -*- coding: utf-8 -*-


!pip install transformers

"""

!wget -O data.csv "https://figshare.com/ndownloader/files/4988956"
!pip install emoji

import nltk
nltk.download('punkt')

!wget -O data.csv "https://figshare.com/ndownloader/files/4988956"
!pip install emoji

import nltk
nltk.download('punkt')

"""## Task 1. Data Cleaning, Preprocessing, and splitting 
The `data` environment contains the SMILE dataset loaded into a pandas dataframe object. Our dataset has three columns: id, tweet, and label. The `tweet` column contains the raw scraped tweet and the `label` column contains the annotated emotion category. Each tweet is labelled with one of the following emotion labels:
- 'nocode', 'not-relevant'
- 'happy', 'happy|surprise', 'happy|sad'
- 'angry', 'disgust|angry', 'disgust'
- 'sad', 'sad|disgust', 'sad|disgust|angry'
- 'surprise'

### Task 1a. Label Consolidation
As we can see above the annotated categories are complex. Several tweets express complex emotions like (e.g. 'happy|sad') or multiple emotions (e.g. 'sad|disgust|angry'). The first things we need to do is clean up our dataset by removing complex examples and consolidating others so that we have a clean set of emotions to predict.

For Task 1a., write code which does the following:
1. Drops all rows which have the label "happy|sad", "happy|surprise", 'sad|disgust|angry', and 'sad|angry'.
2. Re-label 'nocode' and 'not-relevant' as 'no-emotion'.
3. Re-label 'disgust|angry' and 'disgust' as 'angry'.
4. Re-label 'sad|disgust' as 'sad'.

Your updated `data' dataframe should have 3,062 rows and 5 label categories (no-emotion, happy, angry, sad, and surprise).

"""

import pandas as pd

# Refrence- https://note.nkmk.me/en/python-pandas-dataframe-rename/#:~:text=You%20can%20use%20the%20rename,change%20column%2Findex%20name%20individually.&text=Specify%20the%20original%20name%20and,is%20for%20the%20index%20name.
#Refrence- https://www.statology.org/pandas-drop-rows-with-value/
#Refrence- https://www.w3schools.com/python/pandas/ref_df_replace.asp#:~:text=The%20replace()%20method%20replaces,case%20of%20the%20specified%20value.

#Reading CSV
data= pd.read_csv("data.csv")

#Duplicating the row
data=pd.DataFrame([['611857364396965889', '@aandraous @britishmuseum @AndrewsAntonio Merci pour le partage! @openwinemap', 'nocode']], columns=data.columns).append(data)

#Adding the Column Labels
data=data.rename(columns={'611857364396965889': 'Id', '@aandraous @britishmuseum @AndrewsAntonio Merci pour le partage! @openwinemap': 'tweet', 'nocode':'label'})

# Dropping the rows
values = ["happy|sad", "happy|surprise", 'sad|disgust|angry','sad|angry']
data = data[data.label.isin(values) == False]

#Replacing the values
data = data.replace(['nocode','not-relevant','disgust|angry','disgust','sad|disgust'],['no-emotion','no-emotion','angry','angry','sad'])

data

"""### Task 1a Tests
Run the cell below to evaluate your code. To get full credit for this task, your code must pass all tests. Any alteration of the testing code will automatically result in 0 points.
"""

# Test 1. Data should have 5 unique labels.
print(f"Unique label test: {len(data['label'].unique()) == 5}")

# Test 2. Data labels must be: angry, happy, no-emotion, sad, and surprise
labels = ["angry", "happy", "no-emotion", "sad", "surprise"]
print(f"Label check: { set(data['label'].unique()).difference(labels) == set() }")

# Test 3. Check example counts per label
print(f"Angry example count: {len(data[data['label']=='angry']) == 70}")
print(f"Happy example count: {len(data[data['label']=='happy']) == 1137}")
print(f"No-Emotion example count: {len(data[data['label']=='no-emotion']) == 1786}")
print(f"Sad example count: {len(data[data['label']=='sad']) == 34}")
print(f"Surprise example count: {len(data[data['label']=='surprise']) == 35}")

"""### Task 1b. Tweet Cleaning and Processing
Raw tweets are noisy. Consider the example below:
```
'@tateliverpool #BobandRoberta: I am angry more artists that have a profile are not speaking up #foundationcourses. 😠'
```
The mention @tateliverpool and hashtag #BobandRoberta are extra noise that don't directly help with understanding the emotion of the text. The accompanying emoji can be useful but needs to be decoded to it text form :angry: first.

For this task you will fill complete the `preprocess_tweet` function below with the following preprocessing steps:
1. Lower case all text
2. De-emoji the text
3. Remove all hashtags, mentions, and urls
4. Remove all non-alphabet characters except the followng punctuations: period, exclamation mark, and question mark

Hints:
- For step 2 (de-emoji), consider using the python [emoji](https://carpedm20.github.io/emoji/docs/) library. The `emoji.demojize` method will convert all emojis to plain text. The `emoji` library is installed in cell [52].
- Follow the processing steps in order. For example calling nltk's word_tokenize before removing hashtags and mentions will end up creating seperate tokens for @ and # and cause problems.

To get full credit for this task, the Test 1b must pass. Only modify the  cell containing the `preprocess_tweet` function and do not alter the testing code block.

After you are satisfied with your code, run the tests. code to ensure your function works as expected. This cell will also create a new column called `cleaned_tweet` and apply the `preproces_tweet` function to all the examples in the dataset.
"""

#Refrence- https://catriscode.com/2021/03/02/extracting-or-removing-mentions-and-hashtags-in-tweets-using-python/
#Refrence- https://stackoverflow.com/questions/11331982/how-to-remove-any-url-within-a-string-in-python

import emoji
import re

def preprocess_tweet(tweet: str) -> str:
  """
  Function takes a raw tweet and performs the following processing steps:
  1. Lower case all text
  2. De-emoji the text
  3. Remove all hashtags, mentions, and urls
  4. Remove all non-alphabet characters except the followng punctuations: period, exclamation mark, and question mark
  """
  # 1. Lower case all text
  tweet=tweet.lower()

  # 2. De-emoji the text
  tweet=emoji.demojize(tweet, delimiters=("",""))

  # 3. Remove all hashtags, mentions, and urls
  tweet=re.sub("@[A-Za-z0-9_]+","", tweet) # Removing Mentions
  tweet=re.sub("#[A-Za-z0-9_]+","", tweet) # Removing hashtag
  tweet=re.sub(r'http\S+', '', tweet) # Removing urls

  # 4. Remove all non-alphabet characters except the followng punctuations: period, exclamation mark, and question mark
  tweet= re.sub(r'[^\w\s\.\?\!]','',tweet)
  tweet=tweet.replace('_','')

  #5. Removing Awkward spaces
  tweet=tweet.split()
  tweet=" ".join(tweet)

  return tweet


test_tweet = "'@tateliverpool #BobandRoberta: I am angry more artists that have a profile are not speaking up! #foundationcourses 😠'"
print(preprocess_tweet(test_tweet))

"""### Task 1b Test
Run the cell below to evaluate your code. To get full credit for this task, your code must pass all tests. Any alteration of the testing code will automatically result in 0 points.
"""

# Do NOT modify the code below.
# Create new column with cleaned tweets. We will use this for the subsequent tasks
data["cleaned_tweet"] = data["tweet"].apply(preprocess_tweet)

# Test 1b
test_tweet = "'@tateliverpool #BobandRoberta: I am angry more artists that have a profile are not speaking up! #foundationcourses 😠'"
clean_tweet = "i am angry more artists that have a profile are not speaking up! angryface"
print(f"Test 1b: {preprocess_tweet(test_tweet) == clean_tweet}")

"""### Task 1c. Generating Evaluation Splits
Finally, we need to split our data into a train, validation, and test set. We will split the data using a 60-20-20 split, where 60% of our data is used for training, 20% for validation, and 20% for testing. As the dataset is heaviliy imbalanced, make sure you stratify the dataset to ensure that the label distributions across the three splits are roughly equal.

Store your splits in the variables `train`, `val`, and `test` respectively.

Hints:
- Use the [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) function for this task. You'll have to call it twice to get the validation split.
- Set the random state so the sampling can be reproduced (we use 2023 for our random state)
- Use the `stratify` parameter to ensure representative label distributions across the splits.
"""

from sklearn.model_selection import train_test_split

# Your code here
# data_y=data['label']
# data_x=data[['Id','tweet']]

train, temp = train_test_split(data, train_size=0.6,random_state=2023,stratify=data['label'])
val,test=train_test_split(temp, train_size=0.5,random_state=2023,stratify=temp['label'])

"""## Task 2: Naive Baseline Using a Rule-based Classifier 

Now that we have a dataset, let's work on developing some solutions for emotion classification. We'll start with implementing a simple rule-based classifier which will also serve as our naive baseline. Emotive language (e.g. awesome, feel great, super happy) can be a strong signal as to the overall emotion being by the tweet. For each emotion in our label space (happy, surprised, sad, angry) we will generate a set of words and phrases that are often associated with that emotion. At classification time, the classifier will calculate a score based on the overlap between the words in the tweet and the emotive words and phrases for each of the emotions. The emotion label with the highest overlap will be selected as the prediction and if there is no match the "no-emotion" label will be predicted. We can break the implementation of this rules-based classifier into three steps:
1. Emotive language extraction from train examples
2. Developing a scoring algorithm
3. Building the end-to-end classification flow

### Task 2a. Emotive Language Extraction 
For this task you will generate a set of unigrams and bigrams that will be used to predict each of the labels. Using the training data you will need to extract all the unique unigrams and bigrams associated with each label (excluding no-emotion). Then you should ensure that the extracted terms for each emotion label do not appear in the other lists. In the real world, you would then manually curate the generated lists to ensure that associated words were useful and emotive. For the assignment, you won't be required to further curate the generated lists.

Once you've identified the appropiate terms, save them as lists stored in the following environment variables: `happy_words`, `surprised_words`, `sad_words`,and `angry_words`. To get full credit for this section, ensure all 2a Tests pass.

Hints
- We suggest you use Python's [set methods](https://realpython.com/python-sets/) for this task.
- NLTK has a function for extracting [ngrams](https://www.nltk.org/api/nltk.util.html?highlight=ngrams#nltk.util.ngrams). This function expects a list of tokens as input and will output tuples which you'll need to reconvert into strings.
"""

#Refrence- https://blog.finxter.com/how-to-convert-a-list-of-lists-to-a-set-in-python/
# Your code here
from typing import List
from nltk.util import ngrams
from nltk.util import everygrams

# 1. Extract all terms associated with each label
def extract_words(examples: List[str]) -> List[str]:
  """
  Given a list of tweets, return back the unigrams and bigrams found
  across all the tweets.
  """
  extracted_words=(set(everygrams(examples.split(), max_len= 2)))

  return extracted_words

extracted_words=None
happy_words = []
sad_words = []
angry_words = []
surprise_words = []

tweets=list(train['cleaned_tweet'])
labels=list(train['label'])

for i in range(len(tweets)):
    if labels[i]=='happy':
      happy_words.append(extract_words(tweets[i]))
    elif labels[i]=='surprise':
      surprise_words.append(extract_words(tweets[i]))
    elif labels[i]=='sad':
      sad_words.append(extract_words(tweets[i]))
    elif labels[i]=='angry':
      angry_words.append(extract_words(tweets[i]))


words = [happy_words, surprise_words, sad_words, angry_words]
for i in range(len(words)):
    words[i] = set(tuple(word) for word in words[i])
happy_words, surprise_words, sad_words, angry_words = words

"""### Task 2a Tests
Run the cell below to evaluate your code. To get full credit for this task, your code must pass all tests. Any alteration of the testing code will automatically result in 0 points.
"""

# Check sets are non-empty
print("Checking sets are not empty: ")
print(f"Happy words count: {len(happy_words)}, {len(happy_words) > 0}")
print(f"Sad words count: {len(sad_words)}, {len(sad_words) > 0}")
print(f"Angry words count: {len(angry_words)}, {len(angry_words) > 0}")
print(f"Surprise words count: {len(surprise_words)}, {len(surprise_words) > 0}")

# Checks sets are disjoint
union1 = sad_words.union(angry_words, surprise_words)
union2 = happy_words.union(surprise_words, angry_words)
union3 = surprise_words.union(happy_words, sad_words)
union4 = angry_words.union(happy_words, sad_words)

print("\nChecking sets are all disjoint:")
print(f"Happy words disjoint: {happy_words.isdisjoint(union1)}")
print(f"Sad words disjoint: {sad_words.isdisjoint(union2)}")
print(f"Angry words disjoint: {angry_words.isdisjoint(union3)}")
print(f"Surprise words disjoint: {surprise_words.isdisjoint(union4)}")

"""### Task 2b. Scoring using set overlaps

Next we will implement to scoring algorithm. Our score will simply be the count of overlapping terms between tweet text and emotive terms. For this task, finish implementing the code below. To get full credit, ensure Test 2b. is successful.
"""

sample_words = {'cat', 'hat', 'mat', 'bowling', 'bat'}
sample_tweet1 = "that cat is super cool sitting on the mat"
sample_tweet2 = "the man in the bowling hat sat on the cat"
sample_tweet3 = "the quick brown fox jumped over the lazy dog"

def score_tweet(samp_tweet,sample_words):
  count=0
  for x in samp_tweet.split():
    if x in sample_words:
      count+=1
  return count

print(f"Test 1: {score_tweet(sample_tweet1, sample_words) == 2}")
print(f"Test 2: {score_tweet(sample_tweet2, sample_words) == 3}")
print(f"Test 3: {score_tweet(sample_tweet3, sample_words) == 0}")

"""### 2c. Rule-based classification
Let put together our rules-based classfication system. Fill out the logic in the `simple_clf`. Given a tweet, `simple_clf` will generate the overlap score
for each of emotion labels and return the emotion label with the highest score. If there is no match amongst the emotions, the classifier will return 'no-emotion'.

To get full credit for this section, your average F1 score most be greater than 0.
"""

def simple_clf(tweet: str) -> str:
  """
  Given a tweet, calculate all the emotion overlap scores.
  Return the emotion label which has the largest score. If
  overlap score is 0, return no-emotion.
  """
  # Your code here
  emotion_list = [happy_words, surprise_words, sad_words, angry_words]
  emotion_names = ['happy', 'sad', 'angry', 'surprise']
  score = 0
  out = None

  for x in range (len(emotion_list)):
    s= score_tweet(tweet, emot(emotion_list[x]))
    if s > score:
      score = s
      out = emotion_names[x]
  if out != None:
    return out
  else:
    return 'no-emotion'

def emot(x):
  return {k for i in x for j in i for k in j}

"""After finishing the above section, let's evaluate our how model did."""

from sklearn.metrics import classification_report

preds = test["cleaned_tweet"].apply(simple_clf)
print(classification_report(test["label"], preds))



import spacy
from tqdm.notebook import tqdm
nlp = spacy.load("en_core_web_sm")

def generate_pos_features(tweet: str) -> str:
  """
  Given a tweet, return the lemmatized tweet augmented
  with POS tags.
  E.g.:
  Input: "cats are super cool."
  output: "cat-NNS be-VBP super-RB cool-JJ .-."
  """
  POS_tag=nlp(tweet)
  out_string=""
  for x in POS_tag:
    out_string+=(str(x.lemma_)+"-"+str(x.tag_)+" ")
  return out_string.rstrip()

sample_tweet = "I hate action movies"
print(generate_pos_features(sample_tweet))

# Once you have the code working above run this cell.
train["tweet_with_pos"] = train["cleaned_tweet"].apply(generate_pos_features)
test["tweet_with_pos"] = test["cleaned_tweet"].apply(generate_pos_features)

"""### Task 3a Tests
Run the cell below to evaluate your code. To get full credit for this task, your code must pass all tests. Any alteration of the testing code will automatically result in 0 points.
"""

sample_texts = [
    ("i am super angry", "I-PRP be-VBP super-RB angry-JJ"),
    ("That movie was great", "that-DT movie-NN be-VBD great-JJ"),
    ("I hate action movies", "I-PRP hate-VBP action-NN movie-NNS")
]
for i, text in enumerate(sample_texts):
  print(f"Test {i+1}: {generate_pos_features(text[0]) == text[1]}")

"""### Task 3b. Model Training [5 points]
Next we will train two seperate RandomForest Classifier models. For this task you will generate two sets of input features using the `TfidfVectorizer`. We generate Tfidf statistic on the`cleaned_tweet` and the `tweet_with_pos` columns.

Once you've generated your features, train two different Random Forest classifiers with the generated features and generate the predictions on the test set for each classifier.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Your code here
tfidf_vect = TfidfVectorizer()

train_tfidf_features1 = tfidf_vect.fit_transform(train['cleaned_tweet'])
test_tfidf_features1 = tfidf_vect.transform(test['cleaned_tweet'])

train_tfidf_features2 = tfidf_vect.fit_transform(train['tweet_with_pos'])
test_tfidf_features2 = tfidf_vect.transform(test['tweet_with_pos'])


############Classifier 1 #########################

RFC_model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
RFC_model_1.fit(train_tfidf_features1, train['label'])

RFC_predict_1 = RFC_model_1.predict(test_tfidf_features1)

RFC_accuracy_1 = accuracy_score(test['label'], RFC_predict_1)
print("Accuracy of classifier 1:", RFC_accuracy_1)


############Classifier 2 #########################

RFC_model_2 = RandomForestClassifier(n_estimators=100, random_state=42)
RFC_model_2.fit(train_tfidf_features2, train['label'])

RFC_predict_2 = RFC_model_2.predict(test_tfidf_features2)


RFC_accuracy_2 = accuracy_score(test['label'], RFC_predict_2)
print("Accuracy of classifier 2:", RFC_accuracy_2)

"""### Task 3c. [2 points]
Generate classification reports for both models. Print the reports below. In a few sentences (no more than 100 words) explain which features were the most effective and why you think that's the case?
"""

from sklearn.metrics import classification_report

# Classification Report for Tfidf features
print("Classification report for TFIDF features:\n")

# Your code here
print(classification_report(test['label'], RFC_predict_1))


# Classfication Report for POS features
print("Classification report for TFIDF w/ POS features:\n")

# Your code here
print(classification_report(test['label'], RFC_predict_2))

"""### Your evaluation here.

## Task 4. Transfer Learning with DistilBERT [10 points]

For this task you will finetune a pretrained language model (DistilBERT) using the huggingface `transformers` library. For this task you will need to:
- Encode the tweets using the BERT tokenizer
- Create pytorch datasets for for the train, val and test datasets
- Finetune the distilbert model for 5 epochs
- Extract predictions from the model's output logits and convert them into the emotion labels.
- Generate a classification report on the predictions.

Ensure you are running the notebook in Google Colab with the gpu runtime enabled for this section.
"""

#Refrence- Advanced NLP Lab 01 Solutions provided on Blackboard

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
from transformers import TrainingArguments
from sklearn.preprocessing import LabelEncoder

# Your Code here

class SentimentDataset(Dataset):

    def __init__(self, encodings: dict):
        self.encodings = encodings

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict:
        e = {k: v[idx] for k,v in self.encodings.items()}
        return e


AT= AutoTokenizer.from_pretrained("distilbert-base-uncased")


le = LabelEncoder()

# Fit the LabelEncoder on the train labels
# tr= le.fit(train["label"])
# vl= le.fit(val["label"])
# ts= le.fit(test["label"])

# Transform the train labels to numerical representations
train_labels =  le.fit_transform(train["label"])

val_labels = le.fit_transform(val["label"])

test_labels = le.fit_transform(test["label"])



# Train Inputs
train_encodings = AT(
    train["cleaned_tweet"].tolist(),
    padding=True,           # pad all inputs to max length
    max_length=128,         # Bert max is 512, we choose 128 due to compute limitations
    return_tensors="pt",    # Return format pytorch tensor
    truncation=True
)
train_encodings["label"] = torch.tensor(train_labels)  # Update train inputs with labels
train_dataset = SentimentDataset(train_encodings)

# Val Inputs
val_encodings = AT(
    val["cleaned_tweet"].tolist(),
    padding=True,           # pad all inputs to max length
    max_length=128,         # Bert max is 512, we choose 128 due to compute limitations
    return_tensors="pt",     # Return format pytorch tensor
    truncation=True
)
val_encodings["label"] = torch.tensor(val_labels)  # Update train inputs with labels
val_dataset = SentimentDataset(val_encodings)


# Test Inputs
test_encodings = AT(
    test["cleaned_tweet"].tolist(),
    padding=True,           # pad all inputs to max length
    max_length=128,         # Bert max is 512, we choose 128 due to compute limitations
    return_tensors="pt",     # Return format pytorch tensor
    truncation=True
)
test_y = test_labels
test_dataset = SentimentDataset(test_encodings)

model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=5)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    lr_scheduler_type='cosine',
    per_device_train_batch_size = 32,
    per_device_eval_batch_size = 32,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

#Refrence- Advanced NLP Lab 03 Solutions provided on Blackboard
import numpy as np
from sklearn.metrics import classification_report

preds = trainer.predict(test_dataset)
print(preds)

preds = le.inverse_transform(np.argmax(preds.predictions, axis=1))
print(classification_report(test["label"].tolist(), preds))


