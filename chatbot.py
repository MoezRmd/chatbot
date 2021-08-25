import random
import json
import pickle
import numpy as np
import nltk
from flask import Flask, render_template, request, jsonify
from langdetect import language_detection
from googletrans import Translator
from spellchecker import SpellChecker



from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

translator = Translator()
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())


words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if w == word:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bag_of_words(sentence)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    loi = intents_json['intents']
    for i in loi:
        if (i['tag']==tag):
            result = random.choice(i['responses'])
            break
    return result

def correct_spell (lang,text):
    spell = SpellChecker(language=lang)
    t=text.split()
    ch=""
    for w in t:
        ch+=spell.correction(w)+" "
    ch=ch[:len(ch)-1]
    
    return ch

def chatbot_response(message):
    lang = language_detection(message)

    if lang != "en":
        try:
            message=correct_spell("fr",message)
            translation = translator.translate(message, src=lang, dest="en")
        except ValueError:
            return("how can i Help you!")
        else:
            ints = predict_class(translation.text)
            res = get_response(ints, intents)
            translation = translator.translate(res, src="en", dest=lang)
            return(translation.text)
    else:
        message=correct_spell("en",message)
        ints = predict_class(message)
        res = get_response(ints, intents)
        return(res)

print("chatbot is running")
print("welcome")
# while True:
#    message = input("")
 #   if len(message) != 0:
#      lang = language_detection(message)
#        if lang != "en":
#            try:
#                translation = translator.translate(message, src=lang, dest="en")
#            except ValueError:
#                print("how can i Help you!")
#            else:
#                ints = predict_class(translation.text)
#                res = get_response(ints, intents)
#                translation = translator.translate(res, src="en", dest=lang)
#                print(translation.text)
#    elif len(message) == 0:
#        ints = predict_class("hello")
#        res = get_response(ints, intents)
#        print(res)


#    else:
#        ints = predict_class(message)
#        res = get_response(ints, intents)
#        print(res)

