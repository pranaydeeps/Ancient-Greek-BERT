
# Ancient Greek BERT

<img src="https://ichef.bbci.co.uk/images/ic/832xn/p02m4gzb.jpg"/>

The first and only available Ancient Greek sub-word BERT model!

State-of-the-art post fine-tuning on Part-of-Speech Tagging and Morphological Analysis.

Pre-trained weights are made available for a standard 12 layer, 768d BERT-base model.

You can also use the model directly on the HuggingFace Model Hub [here](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT)

Please refer to our paper titled: "A Pilot Study for BERT Language Modelling and Morphological Analysis for Ancient and Medieval Greek". In Proceedings of The 5th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2021)

## How to use

Requirements:

```python
pip install transformers
pip install unicodedata
pip install flair
```

Can be directly used from the HuggingFace Model Hub with:


```python
from transformers import AutoTokenizer, AutoModel
tokeniser = AutoTokenizer.from_pretrained("pranaydeeps/Ancient-Greek-BERT")
model = AutoModel.from_pretrained("pranaydeeps/Ancient-Greek-BERT")  
```

## Fine-tuning for POS/Morphological Analysis

- **finetune_pos.py** can be used to finetune the BERT model for POS tagging on your own data. We provide sample files from the Gold standard treebanks, however the full treebanks can't be made available at this time. Please contact the authors for more details.
- Even though the full treebanks aren't made available, we provide a pre-trained POS Tagging model in the directory SuperPeitho-FLAIR-v2, which can directly be used for inference and has an accuracy of ~90 percent on the 3 treebanks available. You can import the pre-trained model in FLAIR with:

```python
from flair.models import SequenceTagger
tagger = SequenceTagger.load('SuperPeitho-FLAIR-v2/final-model.pt')
```

## Training data

The model was initialised from [AUEB NLP Group's Greek BERT](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1)
and subsequently trained on monolingual data from the First1KGreek Project, Perseus Digital Library, PROIEL Treebank and
Gorman's Treebank

## Training and Eval details

Standard de-accentuating and lower-casing for Greek as suggested in [AUEB NLP Group's Greek BERT](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1)
The model was trained on 4 NVIDIA Tesla V100 16GB GPUs for 80 epochs, with a max-seq-len of 512 and results in a perplexity of 4.8 on the held out test set.
It also gives state-of-the-art results when fine-tuned for PoS Tagging and Morphological Analysis on all 3 treebanks averaging >90% accuracy. Please consult our paper or contact [me](mailto:pranaydeep.singh@ugent.be) for further questions!

## Cite

If you end up using Ancient-Greek-BERT in your research, please cite the paper:

```
@inproceedings{ancient-greek-bert,
author = {Singh, Pranaydeep and Rutten, Gorik and Lefever, Els},
title = {A Pilot Study for BERT Language Modelling and Morphological Analysis for Ancient and Medieval Greek},
year = {2021},
booktitle = {The 5th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2021)}
}
```
