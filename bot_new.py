import spacy
from sklearn.metrics.pairwise import cosine_similarity
import chatbot_lib as lib
from chatbot_lib import User
import spotify_connector as sc


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
env_mode = lib.LOG_LVLS[2]

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
users = pickle.load(open('users.pkl', 'rb'))
# yes I know capital should mean constant but it signifies global for 
# this var because its my program and I can do what I want
CURRENT_USER = 0

documents = pickle.load(open('documents.pkl', 'rb'))

INTERCEPT_CONTEXTS = ["input_name", "input_username", "authenticate_spotify"]

def predict_class_tfidf(sentence, corpus, context):
    sentence = re.sub(r'[.?!,:;()\-\n\d]', ' ', sentence.lower())
    test_corpus = corpus + [sentence]

    lib.log("corpus len: {:d}".format(len(test_corpus)), lib.LOG_LVLS[0], env_mode)

    tfidf_docs = tfidf_vectorizer.fit_transform(test_corpus)
    probs = list(cosine_similarity(tfidf_docs[-1], tfidf_docs)[0][:-1])
    lib.log(probs, lib.LOG_LVLS[0], env_mode)

    probs_adj = []
    lib.log('prob search context: {}'.format(context), lib.LOG_LVLS[0], env_mode)
    k = 0
    for i, p in enumerate(probs):
        if (classes[i][1] == "general" or classes[i][1] ==  context):
            probs_adj.append((p , classes[i][0]))
            lib.log('{class_}: {prob}'.format(class_=probs_adj[k][1], prob=probs_adj[k][0]), lib.LOG_LVLS[1], env_mode)
            k += 1

    #TODO Check for similarity threshold to handle unknown input
    #TODO Handle context

    probs_adj.sort(key=lambda x: x[0], reverse=True)
    #lib.log(pred_index, lib.LOG_LVLS[0], env_mode)
    return probs_adj[0][1]


def extract_name(input_text):
    doc = nlp(input_text)

    # NER proved to not work for many names
    for t in doc:
        if t.pos_ == "PROPN":
            return t.text
    return ""


def rand_response(action_intent):
        list_of_intents = intents['intents']
        result = ""
        for i in list_of_intents:
            if(i['tag'] == action_intent):
                # get context and do action
                result = random.choice(i['responses'])
                break
        return result


def do_action(action_intent, context, **kwargs):
    #dangerous but need global write access to process name
    global CURRENT_USER
    post_context = context
    value = kwargs.get('data', "")

    skip_next = False

    # ===============================
    # GREETING HANDLER
    # ===============================
    if (action_intent == "greeting"):
        return post_context, rand_response(action_intent), skip_next
    
    # ===============================
    # GOODBYE HANDLER
    # ===============================
    elif (action_intent == "goodbye"):
        post_context = "quit"
        return post_context, rand_response(action_intent), skip_next
    
    # ===============================
    # THANKS HANDLER
    # ===============================
    elif (action_intent == "thanks"):
        return post_context, rand_response(action_intent), skip_next
    
    # ===============================
    # OPTIONS HANDLER
    # ===============================
    elif (action_intent == "options"):
        return post_context, rand_response(action_intent), skip_next
    
    # ===============================
    # CONNECT_SPOTIFY HANDLER
    # ===============================
    elif (action_intent == "connect_spotify"):
        if(not users[CURRENT_USER].spotify_username):
            #ask for username
            post_context = "input_username"
            res = "Please enter your spotify username:"
        else:
            #pull from spotify
            post_context = "authenticate_spotify"
            res = "Connecting to Spotify and populating data from account {}".format(users[CURRENT_USER].spotify_username)
            skip_next = True
        return post_context, res, skip_next
    
    # ===============================
    # OPTIONS_2 HANDLER
    # ===============================
    elif (action_intent == "options_2"):
        return post_context, rand_response(action_intent), skip_next

    # ===============================
    # INPUT_NAME HANDLER
    # ===============================
    elif (action_intent == "input_name"):
        response = ""
        lib.log("input: {}".format(value), lib.LOG_LVLS[1], env_mode)
        name = extract_name(value.lower())
        if len(name) > 0:
            for i, u in enumerate(users):
                if u.name == name:
                    CURRENT_USER = i
            # user not already exist
            if CURRENT_USER == 0:
                new_user = User(name, "", [], [], [], [], [])
                CURRENT_USER = len(users)
                users.append(new_user)
                response = "Welcome, {name}!".format(name=users[CURRENT_USER].name)
            else:
                response = "Welcome back, {name}!".format(name=users[CURRENT_USER].name)
            post_context = "main"
        else:
            response = "Sorry, I didn't quite get that"
        return post_context, response, skip_next

    # ===============================
    # INPUT_USERNAME HANDLER
    # ===============================
    elif (action_intent == "input_username"):
        response = ""
        lib.log("input: {}".format(value), lib.LOG_LVLS[1], env_mode)
        #hope for the best with username entry
        name = value.lower()
        if len(name) > 0:
            users[CURRENT_USER].spotify_username = name
            post_context = "authenticate_spotify"
            skip_next = True
        else:
            response = "Sorry, I didn't quite get that"
        return post_context, response, skip_next

    # ===============================
    # AUTHENTICATE_SPOTIFY HANDLER
    # ===============================
    elif (action_intent == "authenticate_spotify"):
        users[CURRENT_USER] = sc.hydrate_user(users[CURRENT_USER])
        post_context = "hydrated"
        res = "connected"
        lib.log(users[CURRENT_USER].artist_likes[:10], lib.LOG_LVLS[0], env_mode)
        return post_context, res, skip_next

    # ===============================
    # TOP_ARTISTS HANDLER
    # ===============================
    elif (action_intent == "top_artists"):
        user = users[CURRENT_USER]
        res = "Your top artists are:\n"
        for i,a in enumerate(user.artist_likes[:10]):
            res += "\n{i}. {artist}".format(i = i+1, artist = a)
        res += "\n"

        return post_context, res, skip_next

    # ===============================
    # TOP_TRACKS HANDLER
    # ===============================
    elif (action_intent == "top_tracks"):
        user = users[CURRENT_USER]
        res = "Your top tracks are:\n"
        for i,s in enumerate(user.song_likes[:10]):
            res += "\n{i}. {track}".format(i = i+1, track = s)
        res += "\n"

        return post_context, res, skip_next

    # ===============================
    # LISTENING_PREFERENCES HANDLER
    # ===============================
    elif (action_intent == "listening_preferences"):
        user = users[CURRENT_USER]
        res = "Your top 5 tracks are:\n"
        for i,s in enumerate(user.song_likes[:5]):
            res += "\n{i}. {track}".format(i = i+1, track = s)
        res += "\n\nYour top 5 artists are:\n"
        for i,a in enumerate(user.artist_likes[:5]):
            res += "\n{i}. {artist}".format(i = i+1, artist = a)
        res += "\n"

        return post_context, res, skip_next

    # ===============================
    # RECOMMEND_MUSIC HANDLER
    # ===============================
    elif (action_intent == "recommend_music"):
        user = users[CURRENT_USER]
        
        recs = sc.get_recommendations(user)
        random.shuffle(recs)
        
        res = "\nHere's some songs you might like:"
        for r in recs[:5]:
            res += "\n{song}".format(song = r)

        return post_context, res, skip_next

    # ===============================
    # DEFAULT/UNKNOWN HANDLER
    # ===============================
    else:
        return post_context, "I'm sorry, I'm having trouble understanding.", skip_next


def handle_message(msg, context):
    data = ""
    skip_next = False
    
    if (context in INTERCEPT_CONTEXTS):
        intent = context
        data = msg
    else:
        documents_raw = [d[0] for d in documents]
        intent = predict_class_tfidf(msg, documents_raw, context)

    lib.log("selected: {}".format(intent), lib.LOG_LVLS[1], env_mode)
    post_context, res, skip_next = do_action(intent, context, data = data)
    lib.log(res, lib.LOG_LVLS[2], env_mode)
    return post_context, skip_next


def exit_program():
    # program shut down and save
    pickle.dump(users, open('users.pkl', 'wb'))
    quit()


if __name__ == "__main__":

    for u in users:
        lib.log(u.name, lib.LOG_LVLS[0], env_mode)

    welcome_text = "Hello, I am Orpheus, a Music reccomendation bot. What is your name?"
    lib.log(welcome_text, lib.LOG_LVLS[2], env_mode)
    curr_context = "input_name"

    skip_next = False
    while (curr_context != "quit"):
        msg = ""
        if(not skip_next):
            msg = input(":: ")
        curr_context, skip_next = handle_message(msg, curr_context)
        lib.log("curr_context: {}".format(curr_context), lib.LOG_LVLS[1], env_mode)

    exit_program()
