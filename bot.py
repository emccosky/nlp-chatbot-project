import spacy
from sklearn.metrics.pairwise import cosine_similarity
from chatbot_lib import User

from keras.models import load_model
import random
import json
import os
import re
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# load a model
nlp = spacy.load('en_core_web_md')

exit_flag = False

lemmatizer = WordNetLemmatizer()
model = load_model(os.path.join(os.getcwd(), 'chatbot_model.h5'))
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

documents = pickle.load(open('documents.pkl', 'rb'))


def clean_input(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words
                      if w not in stopwords.words('english') and len(w) > 1]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


def bow(sentence, words):
    # tokenize the input
    sentence_words = clean_input(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
    return(np.array(bag))


def predict_class(sentence, model, context):
    # filter out predictions below a threshold
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    print(results)
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        res_context = classes[r[0]][1]
        if res_context == context or res_context == "general":
            return_list.append(
                {"intent": classes[r[0]][0], "probability": str(r[1]), "context": res_context})
    return return_list


def predict_class_tfidf(sentence, corpus, context):
    sentence = re.sub(r'[.?!,:;()\-\n\d]', ' ', sentence.lower())
    test_corpus = corpus + [sentence]

    print("corpus len: ", len(test_corpus))

    tfidf_docs = tfidf_vectorizer.fit_transform(test_corpus)
    probs = list(cosine_similarity(tfidf_docs[-1], tfidf_docs)[0][:-1])
    print(probs)

    for i, p in enumerate(probs):
        print('{class_}: {prob}'.format(class_=classes[i][0], prob=p))

    #TODO Check for similarity threshold to handle unknown input
    #TODO Handle context

    pred_index = probs.index(max(probs))
    print(pred_index)
    return classes[pred_index]


def extract_name(input_text):
    doc = nlp(input_text)

    # NER proved to not work for many names
    for t in doc:
        if t.pos_ == "PROPN":
            return t.text
    return ""


def do_action(action_intent):

    def rand_response():
        list_of_intents = intents['intents']
        result = ""
        for i in list_of_intents:
            if(i['tag'] == action_intent):
                # get context and do action
                result = random.choice(i['responses'])
                break
        return result

    def one():
        print("one")

    switcher = {
        "greeting": rand_response(),
        "goodbye": rand_response()
    }
    return switcher.get(action_intent, "Invalid Intent")


def handle_message(msg, context):
    #msg_intents = predict_class(msg, model, context)
    # print(msg_intents)
    #intent = msg_intents[0]['intent']
    #selected_context = msg_intents[0]['context']
    documents_raw = [d[0] for d in documents]
    intent = predict_class_tfidf(msg, documents_raw, context)
    print("selected: ", intent[0])
    print(do_action(intent[0]))


def exit_program():
    # program shut down and save
    pickle.dump(users, open('users.pkl', 'wb'))
    quit()


if __name__ == "__main__":
    users = []
    users = pickle.load(open('users.pkl', 'rb'))
    # for u in users:
    #    u.display()
    response = ""
    name = ""
    curr_context = "general"

    current_user = 0
    while(current_user == 0):
        # collect user name on first run
        print("Hello, I am Orpheus, a Music reccomendation bot. What is your name?")
        response = input(":: ")
        name = extract_name(response.lower())
        if len(name) > 0:
            for i, u in enumerate(users):
                if u.name == name:
                    current_user = i
            # user not already exist
            if current_user == 0:
                new_user = User(name, "", [], [], [], [], [])
                current_user = len(users)
                users.append(new_user)
        else:
            print("I didn't quite get that.")

    if current_user == len(users)-1:
        print("Welcome, {name}!".format(name=users[current_user].name))
    else:
        print("Welcome back, {name}!".format(name=users[current_user].name))

    msg = input(":: ")

    while (msg != "exit"):
        handle_message(msg, curr_context)
        msg = input(":: ")

    exit_program()
