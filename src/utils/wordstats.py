import math
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import brown

# Get a list of document IDs in the Brown Corpus
document_ids = brown.fileids()

# Get a list of documents (as strings) from the Brown Corpus
reference_documents = [' '.join(brown.words(fileid)) for fileid in document_ids]

# 1. Flesch-Kincaid Readability Score
def flesch_kincaid(text):
    num_sentences = len(re.findall(r'[.!?]+', text))
    num_words = len(re.findall(r'\w+', text))
    num_syllables = sum([len(re.findall(r'[aeiouy]+', word, re.IGNORECASE)) for word in re.findall(r'\w+', text)])

    if num_sentences == 0 or num_words == 0:
        
        num_sentences=1
        num_words = max(num_words, 1)

    score = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return score

# 2. Gunning Fog Index
def gunning_fog(text):
    num_sentences = len(re.findall(r'[.!?]+', text))
    num_words = len(re.findall(r'\w+', text))
    num_complex_words = len([word for word in re.findall(r'\w+', text) if len(re.findall(r'[aeiouy]+', word, re.IGNORECASE)) >= 3])

    if num_sentences == 0 or num_words == 0:
        num_sentences=1
        num_words = max(num_words, 1)

    score = 0.4 * ((num_words / num_sentences) + 100 * (num_complex_words / num_words))
    return score

# 3. SMOG Index
def smog(text):
    num_sentences = len(re.findall(r'[.!?]+', text))
    num_complex_words = len([word for word in re.findall(r'\w+', text) if len(re.findall(r'[aeiouy]+', word, re.IGNORECASE)) >= 3])

    if num_sentences < 30:
        num_sentences=1
        #num_words = max(num_words, 1)

    score = 1.043 * math.sqrt(num_complex_words * (30 / num_sentences)) + 3.1291
    return score

# 4. Coleman-Liau Index
def coleman_liau(text):
    num_sentences = len(re.findall(r'[.!?]+', text))
    num_words = len(re.findall(r'\w+', text))
    num_characters = len(re.findall(r'\w', text))

    if num_sentences == 0 or num_words == 0:
        num_sentences=1
        num_words = max(num_words, 1)

    score = 0.0588 * (num_characters / num_words * 100) - 0.296 * (num_sentences / num_words * 100) - 15.8
    return score

# 5. LIX and RIX scores
def lix_and_rix(text):
    num_sentences = len(re.findall(r'[.!?]+', text))
    words = re.findall(r'\w+', text)
    num_words = len(words)
    num_long_words = len([word for word in words if len(word) > 6])

    if num_sentences == 0 or num_words == 0:
        num_sentences=1
        num_words = max(num_words, 1)

    lix = num_words / num_sentences + 100 * (num_long_words / num_words)
    rix = num_long_words / num_sentences
    return lix, rix

# 6. Normalized Inverse Document Frequency (NIDF)
def nidf(text):
    words = re.findall(r'\w+', text.lower())
    word_counts = Counter(words)
    N = len(reference_documents)

    def idf(word):
        df = sum(1 for doc in reference_documents if word in doc.lower())
        return math.log(N / (1 + df)) if df > 0 else 0

    nidf_scores = {word: idf(word) for word in word_counts}
    return nidf_scores

# 7. Term Frequency-Inverse Document Frequency (TF-IDF)
def tf_idf(text):
    vectorizer = TfidfVectorizer()
    reference_documents.append(text)
    matrix = vectorizer.fit_transform(reference_documents)
    
    feature_names = vectorizer.get_feature_names()
    tf_idf_values = matrix.toarray()[-1] # Get the last row of the matrix which corresponds to the input text

    tf_idf_scores = dict(zip(feature_names, tf_idf_values))
    return tf_idf_scores