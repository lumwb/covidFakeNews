from flask import Flask, abort, request, jsonify
from flask_restful import Resource, Api
import spacy
import json
import pickle
import os
from collections import Counter
from string import punctuation
from googlesearch import search
from spacy.lang.en.stop_words import STOP_WORDS
from collections import OrderedDict
import numpy as np
import en_core_web_sm
import re
from difflib import SequenceMatcher

app = Flask(__name__)
api = Api(app)
nlp = en_core_web_sm.load()

basedir = os.path.abspath(os.path.dirname(__file__))
BOOMER_FILE_NAME = os.path.join(basedir, 'boomer.txt')

boomerTextList = []


class TextRank4Keyword():
    """Extract keywords from text"""

    def __init__(self):
        self.d = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 10  # iteration steps
        self.node_weight = None  # save keywords and its weight
        self.top_words = []

    def set_stopwords(self, stopwords):
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True

    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences

    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab

    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs

    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())

    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1

        # Get Symmeric matrix
        g = self.symmetrize(g)

        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        # this is ignore the 0 element in norm
        g_norm = np.divide(g, norm, where=norm != 0)

        return g_norm

    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(
            sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            #print(key + ' - ' + str(value))
            self.top_words.append(key)
            if i > number:
                break

    def analyze(self, text,
                candidate_pos=['NOUN', 'PROPN'],
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""

        # Set stop words
        self.set_stopwords(stopwords)

        # Pare text by spaCy
        doc = nlp(text)

        # Filter sentences
        sentences = self.sentence_segment(
            doc, candidate_pos, lower)  # list of list of words

        # Build vocabulary
        vocab = self.get_vocab(sentences)

        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)

        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)

        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]

        self.node_weight = node_weight


def readBoomerText(fileName):
    global boomerTextList
    # open file and read the content in a list
    with open(fileName, 'rb') as filehandle:
        # read the data as binary data stream
        try:
            boomerTextList = pickle.load(filehandle)
        except EOFError:
            boomerTextList = []
    return boomerTextList


def writeBooomerText(fileName, boomerTextList):
    with open(fileName, 'wb') as filehandle:
        # store the data as binary data stream
        pickle.dump(boomerTextList, filehandle)
    return boomerTextList


class boomerText:
    plaintext = ""
    trueCount = 0
    falseCount = 0
    totalSearchCount = 0
    doc = nlp("")

    def __init__(self, plaintext, doc):
        self.plaintext = plaintext
        self.doc = doc

    def incrementTrueCount(self):
        self.trueCount += 1

    def incrementFalseCount(self):
        self.falseCount += 1

    def incrementTotalSearchCount(self):
        self.totalSearchCount += 1

    def countPercentageFake(self):
        if self.trueCount + self.falseCount == 0:
            return 0

        return (self.falseCount * 100) / (self.trueCount + self.falseCount)


@app.route('/checkFakeNews', methods=['POST'])
def checkFakeNews():
    global boomerTextList
    data = request.json
    plainText = data['boomerText']

    newBoomerTextDoc = nlp(plainText)

    boomerTextExists = False
    boomerTextIndex = -1
    # check if boomer text already exists according to spacy equality
    for index in range(len(boomerTextList)):
        if SequenceMatcher(None, boomerTextList[index].plaintext, plainText).ratio() > 0.95:
            # categorise as same boomer
            boomerTextList[index].incrementTotalSearchCount()
            boomerTextExists = True
            boomerTextIndex = index

    if not boomerTextExists:
        # create new boomer text in list
        newBoomerText = boomerText(plainText, newBoomerTextDoc)
        boomerTextList.append(newBoomerText)
        newBoomerText.incrementTotalSearchCount()
        boomerTextIndex = len(boomerTextList) - 1

    currentBoomerText = boomerTextList[boomerTextIndex]

    # call justin API to get links
    text = plainText.lower()
    query = "covid 19 "

    if (len(text) < 75):
        query = query + text
    else:
        tr4w = TextRank4Keyword()
        tr4w.analyze(text, candidate_pos=[
            'NOUN', 'PROPN'], window_size=4, lower=False)
        tr4w.get_keywords(10)

        if len(tr4w.top_words) >= 7:
            for i in range(7):
                query += tr4w.top_words[i] + " "
        else:
            for keyword in tr4w.top_words:
                query += keyword + " "
        temp = re.findall(r'\d+', text)
        for i in temp:
            query += i + " "

    filter_sites = ["myactivesg.com", "healthhub.sg", "gov.sg",
                    "channelnewsasia.com", "straitstimes.com", "todayonline.com",
                    "www.who.int", "reuters.com"]

    verified_sources = []
    verified_counter = 0

    for i in search(query,        # The query you want to run
                    tld='com',  # The top level domain
                    lang='en',  # The language
                    num=10,     # Number of results per page
                    start=0,    # First result to retrieve
                    stop=30,  # Last result to retrieve
                    pause=2.0,  # Lapse between HTTP requests
                    ):
        for j in range(len(filter_sites)):
            if filter_sites[j] in i:
                verified_sources.append(i)
                verified_counter += 1
        if verified_counter >= 5:
            break

    responseData = {
        "query": query,
        "boomerIndex": boomerTextIndex,
        "percentageFakeNews": currentBoomerText.countPercentageFake(),
        "trueCount": currentBoomerText.trueCount,
        "falseCount": currentBoomerText.falseCount,
        "totalSearchCount": currentBoomerText.totalSearchCount,
        "googleLinks": verified_sources
    }
    return jsonify(responseData), 200


@app.route('/updateVote', methods=['POST'])
def updateVote():
    global boomerTextList
    data = request.json
    boomerIndex = data['boomerIndex']
    voteValue = data['voteValue']

    currentBoomerText = boomerTextList[boomerIndex]

    if voteValue:
        # user voted true
        currentBoomerText.incrementTrueCount()
    else:
        # user voted false
        currentBoomerText.incrementFalseCount()

    responseData = {
        "boomerIndex": boomerIndex,
        "percentageFakeNews": currentBoomerText.countPercentageFake(),
        "trueCount": currentBoomerText.trueCount,
        "falseCount": currentBoomerText.falseCount,
    }

    return jsonify(responseData), 200


@app.route('/getBoomerData', methods=['GET'])
def getBoomerData():
    global boomerTextList
    textList = []
    for boomerText in boomerTextList:
        textList.append(boomerText.plaintext)
    return jsonify(textList), 200


@app.route('/saveBoomerData', methods=['POST'])
def saveBoomerData():
    global boomerTextList
    try:
        writeBooomerText(BOOMER_FILE_NAME, boomerTextList)
        return "All good", 200
    except:
        return "Internal server error", 500


@app.route('/loadBoomerData', methods=['POST'])
def loadBoomerData():
    try:
        readBoomerText(BOOMER_FILE_NAME)
        return "All good", 200
    except:
        return "Internal server error", 500


@app.errorhandler(404)
def not_found(e):
    return '', 404
