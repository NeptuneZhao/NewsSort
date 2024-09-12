import jieba
import numpy as np

def read_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding = "utf8") as f:
        for idx, line in enumerate(f):
            word = line.strip("\n")
            vocab[word] = idx

    return vocab

def convert_example(example, vocab, stop_words, is_test = False):
    if is_test:
        text, = example
    else:
        text, label = example

    input_ids = []
    for word in jieba.cut(text):
        if word in vocab and word not in stop_words:
            word_id = vocab[word]
            input_ids.append(word_id)
        elif word in vocab and word in stop_words:
            continue
        elif word not in vocab:
            word_id = vocab["[UNK]"]
            input_ids.append(word_id)

    valid_length = np.array(len(input_ids), dtype = 'int64')
    input_ids = np.array(input_ids, dtype = 'int64')

    if not is_test:
        label = np.array(label, dtype = "int64")
        return input_ids, valid_length, label
    else:
        return input_ids, valid_length


def preprocess_prediction_data(data, tokenizer):
    examples = []
    for text in data:
        ids = tokenizer.encode(text)
        examples.append([ids, len(ids)])
    return examples


def write_results(labels, file_path):
    with open(file_path, "w", encoding = "utf8") as f:
        f.writelines("\n".join(labels))