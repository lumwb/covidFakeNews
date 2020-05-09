from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import en_core_web_sm
from googlesearch import search

nlp = en_core_web_sm.load()

class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight
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
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):
        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
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
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
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
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight

#text = "Pepper - Antidote For Wuhan Virus By Manidar Nadeson, holistic healer Many years ago when Malaysia was hit by the Nipah virus, malaysians indians was not affected.the reason most of them consumed CURRY a tamilan soup that is made from mixed herbs If rasam is with a mild temerature the body will react to heal any kind of viruses. It is encourage to drink with some hot rice for better effects. It was put to trial during the SARS pendemic and the results are amaxing! Preventive is better than cure start today for tomorrow, the Rasam. Avoid consuming meat that is not properly cooked. Rasam the best way forward to cure Wuhan viruses. Forward to your friends and family you care about, and build up our immune system with RASAM!"
#text = "Interesting advice x From member of the Stanford hospital board. This is their feedback for now on Corona virus: The new Coronavirus may not show sign of infection for many days. How can one know if he/she is infected? By the time they have fever and/or cough and go to the hospital, the lung is usually 50% Fibrosis and it’s too late. Taiwan experts provide a simple self-check that we can do every morning. Take a deep breath and hold your breath for more than 10 seconds. If you complete it successfully without coughing, without discomfort, stiffness or tightness, etc., it proves there is no Fibrosis in the lungs, basically indicates no infection. In critical time, please self-check every morning in an environment with clean air. Serious excellent advice by Japanese doctor treating COVID-19 cases: Everyone should ensure your mouth & throat are moist, never dry. Take a few sips of water every 15 seconds."
text = "From another doctor fm a friend :  Just finished attending a web seminar on Covid19 for 800 Spore doctors. Very depressing. 55% to 70% of Covid19 infected are asymtomatic or have minimal symptoms, but can shed the virus for up to 4 weeks. So it means the 'circuit breaker' or even more drastic measures will extend to 6 weeks or even 8 weeks. Latest large scale study, just released 15min ago, shows that hydroxychloroquine has no positive effect in treatment So mask up everybody everywhere you go and stay at home."

text = text.lower()

tr4w = TextRank4Keyword()
tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
tr4w.get_keywords(10)

query = "covid 19 "
for i in range(7):
    query += tr4w.top_words[i] + " " 

filter_sites = ["myactivesg.com", "healthhub.sg","gov.sg",
                "channelnewsasia.com","straitstimes.com","todayonline.com",
                "www.who.int"]

can_trust = []
counter = 0
counter2 = 0

for i in search(query,        # The query you want to run
                tld = 'com',  # The top level domain
                lang = 'en',  # The language
                num = 10,     # Number of results per page
                start = 0,    # First result to retrieve
                stop = None,  # Last result to retrieve
                pause = 2.0,  # Lapse between HTTP requests
               ):
    for j in range(len(filter_sites)):
        if filter_sites[j] in i:
            can_trust.append(i)
            counter += 1
    counter2 += 1
    if counter2 > 20 or counter >= 5:
        break

print(can_trust)


