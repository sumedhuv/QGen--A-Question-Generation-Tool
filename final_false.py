import re
import spacy
import nltk
import textwrap
from nltk.tree import Tree
from allennlp_models.pretrained import load_predictor
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer, util
import scipy.spatial

# Initialize necessary NLP models
nlp = spacy.load("en_core_web_sm")
predictor = load_predictor("structured-prediction-constituency-parser")
GPT2tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
GPT2model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=GPT2tokenizer.eos_token_id)
BERT_model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

def preprocess_text(text):
    return text.rstrip('?:!.,;')

def parse_sentence(sentence):
    parser_output = predictor.predict(sentence=sentence)
    tree_string = parser_output["trees"]
    return Tree.fromstring(tree_string)

def get_flattened(parse_tree):
    if parse_tree is None:
        return None
    sent_str = [" ".join(x.leaves()) for x in list(parse_tree)]
    return " ".join(sent_str)

def get_right_most_VP_or_NP(parse_tree, last_NP=None, last_VP=None):
    if len(parse_tree.leaves()) == 1:
        return last_NP, last_VP
    for subtree in reversed(parse_tree):
        if subtree.label() == "NP":
            last_NP = subtree
        elif subtree.label() == "VP":
            last_VP = subtree
        break
    return get_right_most_VP_or_NP(subtree, last_NP, last_VP)

def get_termination_portion(main_string, sub_string):
    combined_sub_string = sub_string.replace(" ", "")
    main_string_list = main_string.split()
    for i in range(len(main_string_list)):
        check_string = "".join(main_string_list[i:]).replace(" ", "")
        if check_string == combined_sub_string:
            return " ".join(main_string_list[:i])
    return None

def generate_alternate_endings(partial_sentence, model, tokenizer, num_sequences=10):
    input_ids = tokenizer.encode(partial_sentence, return_tensors='tf')
    maximum_length = len(partial_sentence.split()) + 40
    sample_outputs = model.generate(
        input_ids,
        do_sample=True,
        max_length=maximum_length,
        top_p=0.80,
        top_k=30,
        repetition_penalty=10.0,
        num_return_sequences=num_sequences
    )
    return [tokenizer.decode(sample_output, skip_special_tokens=True) for sample_output in sample_outputs]

def get_dissimilar_sentences(original_sentence, generated_sentences, model):
    original_embedding = model.encode([original_sentence])
    generated_embeddings = model.encode(generated_sentences)
    distances = scipy.spatial.distance.cdist(original_embedding, generated_embeddings, "cosine")[0]
    results = sorted(enumerate(distances), key=lambda x: x[1])
    return [generated_sentences[idx] for idx, _ in results]

# Example usage
test_sentence = "The old woman was sitting under a tree and sipping coffee."
test_sentence = preprocess_text(test_sentence)
tree = parse_sentence(test_sentence)
last_nounphrase, last_verbphrase = get_right_most_VP_or_NP(tree)
longest_phrase = max(get_flattened(last_nounphrase), get_flattened(last_verbphrase), key=len)
split_sentence = get_termination_portion(test_sentence, longest_phrase)

generated_sentences = generate_alternate_endings(split_sentence, GPT2model, GPT2tokenizer)
dissimilar_sentences = get_dissimilar_sentences(test_sentence, generated_sentences, BERT_model)

for sentence in dissimilar_sentences:
    print(sentence)
