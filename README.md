
# Ancient Greek BERT

![](https://media.nationalgeographic.org/assets/photos/141/591/6a85829d-16e1-4392-be49-74e0461e77ec_c0-713-4188-2078_r230x75.JPG?55192c09e6fa5ad5cce49f0b30f7fc05b6c6cb9e)

The first and only available Ancient Greek sub-word BERT model!

State-of-the-art post fine-tuning on Part-of-Speech Tagging and Morphological Analysis.

Pre-trained weights are made available for a standard 12 layer, 768d BERT-base model.

You can also use the model directly on the HuggingFace Model Hub [here](https://huggingface.co/pranaydeeps/Ancient-Greek-BERT)

Please refer to our paper titled: "A Pilot Study for BERT Language Modelling and Morphological Analysis for Ancient and Medieval Greek". In Proceedings of The 5th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2021)

#### How to use

Requirements:

```
pip install transformers
pip install unicodedata
```

Can be directly used from the HuggingFace Model Hub with:


```python
from transformers import AutoTokenizer, AutoModel
tokeniser = AutoTokenizer.from_pretrained("pranaydeeps/Ancient-Greek-BERT")
model = AutoModel.from_pretrained("pranaydeeps/Ancient-Greek-BERT")  
```

## Training data

The model was initialised from [AUEB NLP Group's Greek BERT](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1)
and subsequently trained on monolingual data from the First1KGreek Project, Perseus Digital Library, PROIEL Treebank and
Gorman's Treebank

## Training and Eval details

Standard de-accentuating and lower-casing for Greek as suggested in [AUEB NLP Group's Greek BERT](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1)
The model was trained on 4 NVIDIA Tesla V100 16GB GPUs for 80 epochs, with a max-seq-len of 512 and results in a perplexity of 4.8 on the held out test set.
It also gives state-of-the-art results when fine-tuned for PoS Tagging and Morphological Analysis on all 3 treebanks averaging >90% accuracy. Please consult our paper or contact [me](mailto:pranaydeep.singh@ugent.be) for further questions!
