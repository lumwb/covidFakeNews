from flask import Flask, abort, request, jsonify
from flask_restful import Resource, Api
import spacy
import json
import pickle
import os
# import config
from collections import Counter
from string import punctuation
from googlesearch import search
from spacy.lang.en.stop_words import STOP_WORDS
from collections import OrderedDict
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import re
from difflib import SequenceMatcher
from flask_marshmallow import Marshmallow

app = Flask(__name__)

# load database url before calling SQLAlchemy
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
db = SQLAlchemy(app)

# load flask api config
api = Api(app)

# load marshmallow
ma = Marshmallow(app)

# load spacy english library
nlp = spacy.load("en_core_web_sm")


class boomerText(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    plainText = db.Column(db.TEXT, nullable=False)
    trueCount = db.Column(db.Integer, default=0)
    falseCount = db.Column(db.Integer, default=0)
    totalSearchCount = db.Column(db.Integer, default=0)
    # to create plainText hash for smaller storage and faster comparision

    def __init__(self, plainText):
        self.plainText = plainText


class BoomerTextSchema(ma.Schema):
    class Meta:
        # fields to expose
        fields = ("plainText", "trueCount", "falseCount", "totalSearchCount")


boomerTextSchema = BoomerTextSchema()


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


def getVerifiedLinks(plainText):
    """Get verified news links based on TextRank4Keyword model
    Returns query, verifiedLinks"""
    # call tr4w API to get links
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

    return query, verified_sources


def countPercentageFake(boomerTextModel):
    """Count percentage fake votes given a boomerText SQL Model"""
    if boomerTextModel.trueCount + boomerTextModel.falseCount == 0:
        return 0

    return (boomerTextModel.falseCount * 100) / (boomerTextModel.trueCount + boomerTextModel.falseCount)


@app.route('/checkFakeNews', methods=['POST'])
def checkFakeNews():
    data = request.json
    plainText = data['boomerText']

    # check if boomerText already exists (perfect text match)
    boomerTextResult = boomerText.query.filter_by(
        plainText=plainText).first()

    if boomerTextResult is None:
        # boomerText does not exist
        boomerTextResult = boomerText(plainText)
        boomerTextResult.totalSearchCount = 1
        db.session.add(boomerTextResult)
        db.session.commit()
    else:
        # boomerText already exists
        boomerTextResult.totalSearchCount += 1
        db.session.commit()

    querySearched, verifiedSources = getVerifiedLinks(plainText)

    responseData = {
        "query": querySearched,
        "boomerIndex": boomerTextResult.id,
        "percentageFakeNews": countPercentageFake(boomerTextResult),
        "trueCount": boomerTextResult.trueCount,
        "falseCount": boomerTextResult.falseCount,
        "totalSearchCount": boomerTextResult.totalSearchCount,
        "googleLinks": verifiedSources
    }
    return jsonify(responseData), 200


@app.route('/updateVote', methods=['POST'])
def updateVote():
    data = request.json
    boomerIndex = data['boomerIndex']
    voteValue = data['voteValue']

    # get said boomerText model
    currentBoomerText = boomerText.query.get(boomerIndex)

    if voteValue:
        # user voted true
        currentBoomerText.trueCount += 1
    else:
        # user voted false
        currentBoomerText.falseCount += 1

    # commit changes
    db.session.commit()

    responseData = {
        "boomerIndex": boomerIndex,
        "percentageFakeNews": countPercentageFake(currentBoomerText),
        "trueCount": currentBoomerText.trueCount,
        "falseCount": currentBoomerText.falseCount,
    }

    return jsonify(responseData), 200


@app.route('/getBoomerData', methods=['GET'])
def getBoomerData():
    allBoomerTexts = boomerText.query.all()
    return boomerTextSchema.dumps(allBoomerTexts)


@app.route('/')
def index():
    # Backend server landing page
    basicHtmlPage = """
        <h1>Welcome to covid fake news backend server.</h1>
        <a href="https://github.com/lumwb/covidFakeNews">Click here to visit our Github page</a>
    """
    return basicHtmlPage


@app.errorhandler(404)
def not_found(e):
    return '', 404


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
