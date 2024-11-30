from datasets import load_metric
from itertools import chain
from typing import Callable, Iterable, List, Tuple
from nltk.corpus import WordNetCorpusReader, wordnet
from nltk.stem.api import StemmerI
from nltk.stem.porter import PorterStemmer
import pandas as pd
from datasets import load_metric
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer





# Load BLEU metric
bleu = load_metric("bleu")

# Read predicted and ground truth sentences from text files
def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip().split() for line in file]

predicted_sentences = read_sentences("C:\\Users\\abhij\\OneDrive\\Desktop\\GAN - Research\\Final Phase 2 Implementation and Docs\\GEN.txt")
reference_sentences = read_sentences("C:\\Users\\abhij\\OneDrive\\Desktop\\GAN - Research\\Final Phase 2 Implementation and Docs\\TRU.txt")

print(len(predicted_sentences), len(reference_sentences))
# Ensure the lengths of predicted and reference sentences are the same
assert len(predicted_sentences) == len(reference_sentences)

# Convert the sentences into the required format
references = [[ref] for ref in reference_sentences]

# Evaluate BLEU score for each sentence pair
precision_scores = []
for prediction, reference in zip(predicted_sentences, references):
    # Compute BLEU score for each sentence pair
    bleu_score = bleu.compute(predictions=[prediction], references=[reference])
    # Extract precision scores from BLEU computation
    precision_scores.append(bleu_score['precisions'])


sums = [0] * len(precision_scores[0])

# Calculate sum of each element position across all lists
for precision_list in precision_scores:
    for i, precision in enumerate(precision_list):
        sums[i] += precision

# Calculate average of each element position
num_lists = len(precision_scores)
averages = [total / num_lists for total in sums]

print('BLEU Score', averages)
print()





# Load Rouge metric
rouge = load_metric("rouge")

# Read sentences from text files
def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file]


predicted_sentences = read_sentences("C:\\Users\\abhij\\OneDrive\\Desktop\\ICCIDS Final Preparation\\evaluation\\GEN.txt")
reference_sentences = read_sentences("C:\\Users\\abhij\\OneDrive\\Desktop\\ICCIDS Final Preparation\\evaluation\\TRU.txt")

# Ensure the lengths of predicted and reference sentences are the same
assert len(predicted_sentences) == len(reference_sentences)

# Initialize accumulators for precision, recall, and F-measure
precision_sum = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
recall_sum = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
fmeasure_sum = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
num_pairs = len(predicted_sentences)

# Evaluate Rouge scores for each sentence pair and accumulate the scores
for prediction, reference in zip(predicted_sentences, reference_sentences):
    rouge_score = rouge.compute(predictions=[prediction], references=[[reference]])
    for metric in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']:
        precision_sum[metric] += rouge_score[metric].mid.precision
        recall_sum[metric] += rouge_score[metric].mid.recall
        fmeasure_sum[metric] += rouge_score[metric].mid.fmeasure

# Calculate averages for precision, recall, and F-measure
precision_avg = {metric: precision_sum[metric] / num_pairs for metric in precision_sum}
recall_avg = {metric: recall_sum[metric] / num_pairs for metric in recall_sum}
fmeasure_avg = {metric: fmeasure_sum[metric] / num_pairs for metric in fmeasure_sum}

# Create a DataFrame for the averages
df = pd.DataFrame({
    'Rouge1': [precision_avg['rouge1'], recall_avg['rouge1'], fmeasure_avg['rouge1']],
    'Rouge2': [precision_avg['rouge2'], recall_avg['rouge2'], fmeasure_avg['rouge2']],
    'RougeL': [precision_avg['rougeL'], recall_avg['rougeL'], fmeasure_avg['rougeL']],
    'RougeLsum': [precision_avg['rougeLsum'], recall_avg['rougeLsum'], fmeasure_avg['rougeLsum']]
}, index=['Precision', 'Recall', 'F-measure'])

print(df)
print()





def _generate_enums(
    hypothesis: Iterable[str],
    reference: Iterable[str],
    preprocess: Callable[[str], str] = str.lower,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]]]:
    enum_hypothesis_list = list(enumerate(map(preprocess, hypothesis)))
    enum_reference_list = list(enumerate(map(preprocess, reference)))
    return enum_hypothesis_list, enum_reference_list

def exact_match(
    hypothesis: Iterable[str], reference: Iterable[str]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _match_enums(enum_hypothesis_list, enum_reference_list)

def _match_enums(
    enum_hypothesis_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        for j in range(len(enum_reference_list))[::-1]:
            if enum_hypothesis_list[i][1] == enum_reference_list[j][1]:
                word_match.append((enum_hypothesis_list[i][0], enum_reference_list[j][0]))
                enum_hypothesis_list.pop(i)
                enum_reference_list.pop(j)
                break
    return word_match, enum_hypothesis_list, enum_reference_list

def _enum_stem_match(
    enum_hypothesis_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
    stemmer: StemmerI = PorterStemmer(),
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    stemmed_enum_hypothesis_list = [(word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_hypothesis_list]
    stemmed_enum_reference_list = [(word_pair[0], stemmer.stem(word_pair[1])) for word_pair in enum_reference_list]
    return _match_enums(stemmed_enum_hypothesis_list, stemmed_enum_reference_list)

def stem_match(
    hypothesis: Iterable[str],
    reference: Iterable[str],
    stemmer: StemmerI = PorterStemmer(),
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_stem_match(enum_hypothesis_list, enum_reference_list, stemmer=stemmer)

def _enum_wordnetsyn_match(
    enum_hypothesis_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    word_match = []
    for i in range(len(enum_hypothesis_list))[::-1]:
        hypothesis_syns = set(chain.from_iterable(
            (lemma.name() for lemma in synset.lemmas() if lemma.name().find("_") < 0)
            for synset in wordnet.synsets(enum_hypothesis_list[i][1])
        )).union({enum_hypothesis_list[i][1]})
        for j in range(len(enum_reference_list))[::-1]:
            if enum_reference_list[j][1] in hypothesis_syns:
                word_match.append((enum_hypothesis_list[i][0], enum_reference_list[j][0]))
                enum_hypothesis_list.pop(i)
                enum_reference_list.pop(j)
                break
    return word_match, enum_hypothesis_list, enum_reference_list

def wordnetsyn_match(
    hypothesis: Iterable[str],
    reference: Iterable[str],
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_wordnetsyn_match(enum_hypothesis_list, enum_reference_list, wordnet=wordnet)

def _enum_align_words(
    enum_hypothesis_list: List[Tuple[int, str]],
    enum_reference_list: List[Tuple[int, str]],
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    exact_matches, enum_hypothesis_list, enum_reference_list = _match_enums(
        enum_hypothesis_list, enum_reference_list
    )
    stem_matches, enum_hypothesis_list, enum_reference_list = _enum_stem_match(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer
    )
    wns_matches, enum_hypothesis_list, enum_reference_list = _enum_wordnetsyn_match(
        enum_hypothesis_list, enum_reference_list, wordnet=wordnet
    )
    return (
        sorted(
            exact_matches + stem_matches + wns_matches, key=lambda wordpair: wordpair[0]
        ),
        enum_hypothesis_list,
        enum_reference_list,
    )

def align_words(
    hypothesis: Iterable[str],
    reference: Iterable[str],
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    enum_hypothesis_list, enum_reference_list = _generate_enums(hypothesis, reference)
    return _enum_align_words(
        enum_hypothesis_list, enum_reference_list, stemmer=stemmer, wordnet=wordnet
    )

def _count_chunks(matches: List[Tuple[int, int]]) -> int:
    i = 0
    chunks = 1
    while i < len(matches) - 1:
        if (matches[i + 1][0] == matches[i][0] + 1) and (
            matches[i + 1][1] == matches[i][1] + 1
        ):
            i += 1
            continue
        i += 1
        chunks += 1
    return chunks

def single_meteor_score(
    reference: Iterable[str],
    hypothesis: Iterable[str],
    preprocess: Callable[[str], str] = str.lower,
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    enum_hypothesis, enum_reference = _generate_enums(
        hypothesis, reference, preprocess=preprocess
    )
    translation_length = len(enum_hypothesis)
    reference_length = len(enum_reference)
    matches, _, _ = _enum_align_words(
        enum_hypothesis, enum_reference, stemmer=stemmer, wordnet=wordnet
    )
    matches_count = len(matches)
    try:
        precision = float(matches_count) / translation_length
        recall = float(matches_count) / reference_length
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        chunk_count = float(_count_chunks(matches))
        frag_frac = chunk_count / matches_count
    except ZeroDivisionError:
        return 0.0
    penalty = gamma * frag_frac**beta
    return (1 - penalty) * fmean

def meteor_score(
    references: Iterable[Iterable[str]],
    hypothesis: Iterable[str],
    preprocess: Callable[[str], str] = str.lower,
    stemmer: StemmerI = PorterStemmer(),
    wordnet: WordNetCorpusReader = wordnet,
    alpha: float = 0.9,
    beta: float = 3.0,
    gamma: float = 0.5,
) -> float:
    return max(
        single_meteor_score(
            reference,
            hypothesis,
            preprocess=preprocess,
            stemmer=stemmer,
            wordnet=wordnet,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )
        for reference in references
    )






# Read sentences from text files
def read_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip().split() for line in file]

predicted_sentences = read_sentences("C:\\Users\\abhij\\OneDrive\\Desktop\\ICCIDS Final Preparation\\evaluation\\GEN.txt")
reference_sentences = read_sentences("C:\\Users\\abhij\\OneDrive\\Desktop\\ICCIDS Final Preparation\\evaluation\\TRU.txt")

# Ensure the lengths of predicted and reference sentences are the same
assert len(predicted_sentences) == len(reference_sentences)

# Initialize METEOR score accumulator
meteor_scores = []

# Initialize stemmer
stemmer = PorterStemmer()

# Compute METEOR score for each sentence pair
for prediction, reference in zip(predicted_sentences, reference_sentences):
    score = meteor_score([reference], prediction, stemmer=stemmer, wordnet=wordnet, alpha=0.9, beta=3.0, gamma=0.5)
    meteor_scores.append(score)