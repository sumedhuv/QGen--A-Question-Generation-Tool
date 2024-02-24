import nltk
import pke
import re
import string
from flashtext import KeywordProcessor
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from pprint import pprint
import traceback

# Ensure necessary NLTK data packages are downloaded.
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

text = """The culture of nuclear families is in fashion. Parents are often heard complaining about the difficulties in bringing up children these days. Too much of freedom in demand, too much independence; over night parties; excessive extravagance, splurging pocket money; no time for studies and family all this is a common cry of such families. Aren’t parents, themselves, responsible for this pitiful state ? The basic need of a growing youth is the family, love, attention and bonding along with moral values. One should not forget that ‘charity begins at home’.

Independence and individuality both need to be respected, in order to maintain the sanctity of family. Children, today are to be handled with tact in order to bridge the ever widening generation gap. Only the reasonable demands need to be fulfilled, as there are too many expenses to be met and top many social obligations to be taken care of by the parents. Our forefathers lived happily in joint families. Children loved to live with their cousins, learnt to adjust within means. There was perfect harmony between the generations. There never existed the concept of old-age homes. There was deep respect for the family elders and love, care and concern for the youngsters. Even the minor family differences were solved amicably."""

def tokenize_sentences(text):
    """Tokenize the provided text into sentences, filtering out short sentences."""
    sentences = sent_tokenize(text)
    return [sentence.strip() for sentence in sentences if len(sentence) > 20]

def get_noun_adj_verb(text):
    """Extract key phrases (nouns, adjectives, verbs) from the text."""
    keyphrases = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=text, language='en')
        pos = {'NOUN', 'ADJ', 'VERB'}
        # Note: The candidate_selection method does not accept a stoplist argument.
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = [val[0] for val in extractor.get_n_best(n=30)]
    except Exception as e:
        traceback.print_exc()
    return keyphrases

def get_sentences_for_keyword(keywords, sentences):
    """Map keywords to sentences they appear in."""
    keyword_processor = KeywordProcessor()
    keyword_sentences = {word: [] for word in keywords}
    for word in keywords:
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        found_keywords = keyword_processor.extract_keywords(sentence)
        for key in found_keywords:
            keyword_sentences[key].append(sentence)
    # Sort sentences by length in descending order for each keyword
    for key, values in keyword_sentences.items():
        keyword_sentences[key] = sorted(values, key=len, reverse=True)
    return keyword_sentences

def get_fill_in_the_blanks(sentence_mapping):
    """Generate fill-in-the-blank sentences from the sentence mapping."""
    output = {"title": "Fill in the blanks for these sentences with matching words at the top"}
    blank_sentences, processed, keys = [], [], []
    for key, sentences in sentence_mapping.items():
        if sentences:
            sentence = sentences[0]
            pattern = re.compile(re.escape(key), re.IGNORECASE)
            if len(re.findall(pattern, sentence)) < 2 and sentence not in processed:
                processed.append(sentence)
                keys.append(key)
                blank_sentences.append(pattern.sub(" _________ ", sentence))
    output["sentences"] = blank_sentences[:10]
    output["keys"] = keys[:10]
    return output

# Processing flow
sentences = tokenize_sentences(text)
keywords = get_noun_adj_verb(text)
keyword_sentence_mapping = get_sentences_for_keyword(keywords, sentences)
fill_in_the_blanks = get_fill_in_the_blanks(keyword_sentence_mapping)
pprint(fill_in_the_blanks)
