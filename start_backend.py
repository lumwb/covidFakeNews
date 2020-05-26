import time
import sys
import random
from flask import Flask, abort, request, jsonify
from flask_restful import Resource, Api
import spacy
import json
import pickle
import os
# import config
from collections import Counter
from string import punctuation
# from googlesearch import search
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
# app.config['SQLALCHEMY_DATABASE_URI'] = config.PSQL_URL
db = SQLAlchemy(app)

# load flask api config
api = Api(app)

# load marshmallow
ma = Marshmallow(app)

# load spacy english library
nlp = spacy.load("en_core_web_sm")

URL_GENERATED = ""

if sys.version_info[0] > 2:
    from http.cookiejar import LWPCookieJar
    from urllib.request import Request, urlopen
    from urllib.parse import quote_plus, urlparse, parse_qs
else:
    from cookielib import LWPCookieJar
    from urllib import quote_plus
    from urllib2 import Request, urlopen
    from urlparse import urlparse, parse_qs

try:
    from bs4 import BeautifulSoup
    is_bs4 = True
except ImportError:
    from BeautifulSoup import BeautifulSoup
    is_bs4 = False

__all__ = [

    # Main search function.
    'search',

    # Specialized search functions.
    'search_images', 'search_news',
    'search_videos', 'search_shop',
    'search_books', 'search_apps',

    # Shortcut for "get lucky" search.
    'lucky',

    # Miscellaneous utility functions.
    'get_random_user_agent', 'get_tbs',
]

# URL templates to make Google searches.
url_home = "https://www.google.%(tld)s/"
url_search = "https://www.google.%(tld)s/search?hl=%(lang)s&q=%(query)s&" \
             "btnG=Google+Search&tbs=%(tbs)s&safe=%(safe)s&tbm=%(tpe)s&" \
             "cr=%(country)s"
url_next_page = "https://www.google.%(tld)s/search?hl=%(lang)s&q=%(query)s&" \
                "start=%(start)d&tbs=%(tbs)s&safe=%(safe)s&tbm=%(tpe)s&" \
                "cr=%(country)s"
url_search_num = "https://www.google.%(tld)s/search?hl=%(lang)s&q=%(query)s&" \
                 "num=%(num)d&btnG=Google+Search&tbs=%(tbs)s&safe=%(safe)s&" \
                 "tbm=%(tpe)s&cr=%(country)s"
url_next_page_num = "https://www.google.%(tld)s/search?hl=%(lang)s&" \
                    "q=%(query)s&num=%(num)d&start=%(start)d&tbs=%(tbs)s&" \
                    "safe=%(safe)s&tbm=%(tpe)s&cr=%(country)s"
url_parameters = (
    'hl', 'q', 'num', 'btnG', 'start', 'tbs', 'safe', 'tbm', 'cr')

# Cookie jar. Stored at the user's home folder.
# If the cookie jar is inaccessible, the errors are ignored.
home_folder = os.getenv('HOME')
if not home_folder:
    home_folder = os.getenv('USERHOME')
    if not home_folder:
        home_folder = '.'   # Use the current folder on error.
cookie_jar = LWPCookieJar(os.path.join(home_folder, '.google-cookie'))
try:
    cookie_jar.load()
except Exception:
    pass

# Default user agent, unless instructed by the user to change it.
USER_AGENT = 'Mozilla/4.0 (compatible; MSIE 8.0; Windows NT 6.0)'

# Load the list of valid user agents from the install folder.
# The search order is:
#   * user_agents.txt.gz
#   * user_agents.txt
#   * default user agent
try:
    install_folder = os.path.abspath(os.path.split(__file__)[0])
    try:
        user_agents_file = os.path.join(install_folder, 'user_agents.txt.gz')
        import gzip
        fp = gzip.open(user_agents_file, 'rb')
        try:
            user_agents_list = [_.strip() for _ in fp.readlines()]
        finally:
            fp.close()
            del fp
    except Exception:
        user_agents_file = os.path.join(install_folder, 'user_agents.txt')
        with open(user_agents_file) as fp:
            user_agents_list = [_.strip() for _ in fp.readlines()]
except Exception:
    user_agents_list = [USER_AGENT]


# Get a random user agent.
def get_random_user_agent():
    """
    Get a random user agent string.

    :rtype: str
    :return: Random user agent string.
    """
    return random.choice(user_agents_list)


# Helper function to format the tbs parameter.
def get_tbs(from_date, to_date):
    """
    Helper function to format the tbs parameter.

    :param datetime.date from_date: Python date object.
    :param datetime.date to_date: Python date object.

    :rtype: str
    :return: Dates encoded in tbs format.
    """
    from_date = from_date.strftime('%m/%d/%Y')
    to_date = to_date.strftime('%m/%d/%Y')
    return 'cdr:1,cd_min:%(from_date)s,cd_max:%(to_date)s' % vars()


# Request the given URL and return the response page, using the cookie jar.
# If the cookie jar is inaccessible, the errors are ignored.
def get_page(url, user_agent=None):
    """
    Request the given URL and return the response page, using the cookie jar.

    :param str url: URL to retrieve.
    :param str user_agent: User agent for the HTTP requests.
        Use None for the default.

    :rtype: str
    :return: Web page retrieved for the given URL.

    :raises IOError: An exception is raised on error.
    :raises urllib2.URLError: An exception is raised on error.
    :raises urllib2.HTTPError: An exception is raised on error.
    """
    if user_agent is None:
        user_agent = USER_AGENT
    request = Request(url)
    request.add_header('User-Agent', user_agent)
    cookie_jar.add_cookie_header(request)
    response = urlopen(request)
    cookie_jar.extract_cookies(response, request)
    html = response.read()
    response.close()
    try:
        cookie_jar.save()
    except Exception:
        pass
    return html


# Filter links found in the Google result pages HTML code.
# Returns None if the link doesn't yield a valid result.
def filter_result(link):
    try:

        # Decode hidden URLs.
        if link.startswith('/url?'):
            o = urlparse(link, 'http')
            link = parse_qs(o.query)['q'][0]

        # Valid results are absolute URLs not pointing to a Google domain,
        # like images.google.com or googleusercontent.com for example.
        # TODO this could be improved!
        o = urlparse(link, 'http')
        if o.netloc and 'google' not in o.netloc:
            return link

    # On error, return None.
    except Exception:
        pass


# Returns a generator that yields URLs.
def search(query, tld='com', lang='en', tbs='0', safe='off', num=10, start=0,
           stop=None, domains=None, pause=2.0, tpe='', country='',
           extra_params=None, user_agent=None):
    """
    Search the given query string using Google.

    :param str query: Query string. Must NOT be url-encoded.
    :param str tld: Top level domain.
    :param str lang: Language.
    :param str tbs: Time limits (i.e "qdr:h" => last hour,
        "qdr:d" => last 24 hours, "qdr:m" => last month).
    :param str safe: Safe search.
    :param int num: Number of results per page.
    :param int start: First result to retrieve.
    :param int stop: Last result to retrieve.
        Use None to keep searching forever.
    :param list domains: A list of web domains to constrain
        the search.
    :param float pause: Lapse to wait between HTTP requests.
        A lapse too long will make the search slow, but a lapse too short may
        cause Google to block your IP. Your mileage may vary!
    :param str tpe: Search type (images, videos, news, shopping, books, apps)
        Use the following values {videos: 'vid', images: 'isch',
        news: 'nws', shopping: 'shop', books: 'bks', applications: 'app'}
    :param str country: Country or region to focus the search on. Similar to
        changing the TLD, but does not yield exactly the same results.
        Only Google knows why...
    :param dict extra_params: A dictionary of extra HTTP GET
        parameters, which must be URL encoded. For example if you don't want
        Google to filter similar results you can set the extra_params to
        {'filter': '0'} which will append '&filter=0' to every query.
    :param str user_agent: User agent for the HTTP requests.
        Use None for the default.

    :rtype: generator of str
    :return: Generator (iterator) that yields found URLs.
        If the stop parameter is None the iterator will loop forever.
    """
    # Set of hashes for the results found.
    # This is used to avoid repeated results.
    hashes = set()

    # Count the number of links yielded.
    count = 0

    # Prepare domain list if it exists.
    if domains:
        query = query + ' ' + ' OR '.join(
            'site:' + domain for domain in domains)

    # Prepare the search string.
    query = quote_plus(query)

    # If no extra_params is given, create an empty dictionary.
    # We should avoid using an empty dictionary as a default value
    # in a function parameter in Python.
    if not extra_params:
        extra_params = {}

    # Check extra_params for overlapping.
    for builtin_param in url_parameters:
        if builtin_param in extra_params.keys():
            raise ValueError(
                'GET parameter "%s" is overlapping with \
                the built-in GET parameter',
                builtin_param
            )

    # Grab the cookie from the home page.
    get_page(url_home % vars(), user_agent)

    # Prepare the URL of the first request.
    if start:
        if num == 10:
            url = url_next_page % vars()
        else:
            url = url_next_page_num % vars()
    else:
        if num == 10:
            url = url_search % vars()
        else:
            url = url_search_num % vars()

    # Loop until we reach the maximum result, if any (otherwise, loop forever).
    while not stop or count < stop:

        # Remeber last count to detect the end of results.
        last_count = count

        # Append extra GET parameters to the URL.
        # This is done on every iteration because we're
        # rebuilding the entire URL at the end of this loop.
        for k, v in extra_params.items():
            k = quote_plus(k)
            v = quote_plus(v)
            url = url + ('&%s=%s' % (k, v))

        # Sleep between requests.
        # Keeps Google from banning you for making too many requests.
        time.sleep(pause)
        print(url)
        print('TEST DRAGON')
        global URL_GENERATED
        URL_GENERATED = url
        # Request the Google Search results page.
        html = get_page(url, user_agent)

        # Parse the response and get every anchored URL.
        if is_bs4:
            soup = BeautifulSoup(html, 'html.parser')
        else:
            soup = BeautifulSoup(html)
        try:
            anchors = soup.find(id='search').findAll('a')
            # Sometimes (depending on the User-agent) there is
            # no id "search" in html response...
        except AttributeError:
            # Remove links of the top bar.
            gbar = soup.find(id='gbar')
            if gbar:
                gbar.clear()
            anchors = soup.findAll('a')

        # Process every anchored URL.
        for a in anchors:

            # Get the URL from the anchor tag.
            try:
                link = a['href']
            except KeyError:
                continue

            # Filter invalid links and links pointing to Google itself.
            link = filter_result(link)
            if not link:
                continue

            # Discard repeated results.
            h = hash(link)
            if h in hashes:
                continue
            hashes.add(h)

            # Yield the result.
            yield link

            # Increase the results counter.
            # If we reached the limit, stop.
            count += 1
            if stop and count >= stop:
                return

        # End if there are no more results.
        # XXX TODO review this logic, not sure if this is still true!
        if last_count == count:
            break

        # Prepare the URL for the next request.
        start += num
        if num == 10:
            url = url_next_page % vars()
        else:
            url = url_next_page_num % vars()


# Shortcut to search images.
# Beware, this does not return the image link.
def search_images(*args, **kwargs):
    """
    Shortcut to search images.

    Same arguments and return value as the main search function.

    :note: Beware, this does not return the image link.
    """
    kwargs['tpe'] = 'isch'
    return search(*args, **kwargs)


# Shortcut to search news.
def search_news(*args, **kwargs):
    """
    Shortcut to search news.

    Same arguments and return value as the main search function.
    """
    kwargs['tpe'] = 'nws'
    return search(*args, **kwargs)


# Shortcut to search videos.
def search_videos(*args, **kwargs):
    """
    Shortcut to search videos.

    Same arguments and return value as the main search function.
    """
    kwargs['tpe'] = 'vid'
    return search(*args, **kwargs)


# Shortcut to search shop.
def search_shop(*args, **kwargs):
    """
    Shortcut to search shop.

    Same arguments and return value as the main search function.
    """
    kwargs['tpe'] = 'shop'
    return search(*args, **kwargs)


# Shortcut to search books.
def search_books(*args, **kwargs):
    """
    Shortcut to search books.

    Same arguments and return value as the main search function.
    """
    kwargs['tpe'] = 'bks'
    return search(*args, **kwargs)


# Shortcut to search apps.
def search_apps(*args, **kwargs):
    """
    Shortcut to search apps.

    Same arguments and return value as the main search function.
    """
    kwargs['tpe'] = 'app'
    return search(*args, **kwargs)


# Shortcut to single-item search.
# Evaluates the iterator to return the single URL as a string.
def lucky(*args, **kwargs):
    """
    Shortcut to single-item search.

    Same arguments as the main search function, but the return value changes.

    :rtype: str
    :return: URL found by Google.
    """
    return next(search(*args, **kwargs))


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

    text = str.split(text)
    for i in text:
        i = i.replace(".", "")
        i = i.replace(",", "")
        if i.isdigit():
            query += " " + i

    filter_sites = ["myactivesg.com", "healthhub.sg", "gov.sg",
                    "channelnewsasia.com", "straitstimes.com", "todayonline.com",
                    "www.who.int", "reuter.com", "businessinsider.sg"]

    verified_sources = []
    verified_counter = 0

    searchResult = search(query,        # The query you want to run
                          tld='com',  # The top level domain
                          lang='en',  # The language
                          num=10,     # Number of results per page
                          start=0,    # First result to retrieve
                          stop=30,  # Last result to retrieve
                          pause=2.0,  # Lapse between HTTP requests
                          country='SG'
                          #   tpe='nws'
                          )

    for i in searchResult:
        for j in range(len(filter_sites)):
            if (filter_sites[j] in i) and (i not in verified_sources):
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
    global URL_GENERATED
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
        "googleLinks": verifiedSources,
        "urlGenerated": URL_GENERATED
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
