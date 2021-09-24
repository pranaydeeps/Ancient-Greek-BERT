#!/usr/bin/env python
# coding: utf-8
import ast
from sklearn.model_selection import train_test_split
from flair.data import Corpus
from flair.embeddings import TokenEmbeddings, WordEmbeddings, StackedEmbeddings, CharacterEmbeddings, TransformerWordEmbeddings
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer


file = open("data/train.txt", "r")

contents = file.read()
dictionary = ast.literal_eval(contents)

file.close()

file = open("data/test.txt", "r")

contents = file.read()
test_dictionary = ast.literal_eval(contents)

file.close()

file = open("data/dev.txt", "r")

contents = file.read()
dev_dictionary = ast.literal_eval(contents)

file.close()

def convert_dict(dictionary):
    new_format = []
    for item in dictionary:
        words = item['tokens']
        tags = item['pos_tags']
        for i, word in enumerate(words):
            new_format.append("{}\t{}".format(words[i], tags[i]))
        new_format.append("\n")
    return new_format

train_list = convert_dict(dictionary)
test_list = convert_dict(test_dictionary)
dev_list = convert_dict(dev_dictionary)

textfile = open("data/pos_flair_format_train.txt", "w")
for element in train_list:
    textfile.write(element + "\n")
textfile.close()

textfile = open("data/pos_flair_format_test.txt", "w")
for element in test_list:
    textfile.write(element + "\n")
textfile.close()

textfile = open("data/pos_flair_format_dev.txt", "w")
for element in dev_list:
    textfile.write(element + "\n")
textfile.close()
columns = {0: 'text', 1: 'pos'}

data_folder = 'data/'

corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='pos_flair_format_train.txt',
                              dev_file='pos_flair_format_dev.txt',
                              test_file='pos_flair_format_test.txt')

tag_type = 'pos'

tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary)

embedding_types = [
    TransformerWordEmbeddings("pranaydeeps/Ancient-Greek-BERT"),
    CharacterEmbeddings()
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                        embeddings=embeddings,
                                        tag_dictionary=tag_dictionary,
                                        tag_type=tag_type,
                                        use_crf=True)

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train('POS_TAGGER_ANCIENT_GREEK',
              learning_rate=0.1,
              mini_batch_size=8,
              max_epochs=80,
              monitor_train=True,
              monitor_test=True,
              train_with_dev=False)


