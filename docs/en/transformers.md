---
layout: docs
header: true
title: Transformers
permalink: /docs/en/transformers
key: docs-transformers
modify_date: "2021-08-05"
use_language_switcher: "Python-Scala-Java"
---

<script> {% include scripts/approachModelSwitcher.js %} </script>

{% assign parent_path = "en/transformer_entries" %}

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "transformer_entries/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}

## Import Transformers into Spark NLP

### Overview

Spark NLP 🚀  3.1.0 is out! We have extended support for HuggingFace 🤗  exported models in equivalent Spark NLP annotators. Starting this release, you can easily use `saved_model` feature in HuggingFace within a few lines of codes and import any BERT, DistilBERT, RoBERTa, XLM-RoBERTa, Longformer, BertForTokenClassification, and DistilBertForTokenClassification models to Spark NLP. We will work on the remaining annotators and extend this support to the rest with each release 😊

### Compatibility

**Spark NLP**: The equivalent annotator in Spark NLP
**TF Hub**: Models from [TF Hub](https://tfhub.dev/)
**HuggingFace**: Models from [HuggingFace](https://huggingface.co/models)
**Model Architecture**: Which architecture is compatible with that annotator
**Flags**:

- Fully supported ✅
- Partially supported (requires workarounds) ✔️
- Under development ❎
- Not supported ❌

Spark NLP | TF Hub | HuggingFace | Model Architecture
:------------ | :-------------| :-------------| :-------------|
BertEmbeddings |  ✅  |  ✅  |  BERT - Small BERT - ELECTRA
BertSentenceEmbeddings |  ✅  | ✅   | BERT - Small BERT - ELECTRA
DistilBertEmbeddings|   |  ✅   | DistilBERT
RoBertaEmbeddings |   | ✅   | RoBERTa - DistilRoBERTa
XlmRoBertaEmbeddings |   | ✅   | XLM-RoBERTa
AlbertEmbeddings | ✅  |  ✅   |  ALBERT
XlnetEmbeddings |   | ✅  |  XLNet
LongformerEmbeddings |   | ✅  | Longformer
BertForTokenClassification |   | ✅  |  BERT - Small BERT - ELECTRA
DistilBertForTokenClassification |   | ✅  |  DistilBERT
ElmoEmbeddings | ❎  |  ❎  |
UniversalSentenceEncoder |  ❎ |   |
T5Transformer |   |  ❌ |
MarianTransformer|   | ❌  |

### Example Notebooks

#### HuggingFace to Spark NLP

Spark NLP | HuggingFace Notebooks | Colab
:------------ | :-------------| :----------|
BertEmbeddings |  [HuggingFace in Spark NLP - BERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT.ipynb)
BertSentenceEmbeddings | [HuggingFace in Spark NLP - BERT Sentence](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb)
DistilBertEmbeddings| [HuggingFace in Spark NLP - DistilBERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBERT.ipynb)
RoBertaEmbeddings | [HuggingFace in Spark NLP - RoBERTa](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBERTa.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBERTa.ipynb)
XlmRoBertaEmbeddings | [HuggingFace in Spark NLP - XLM-RoBERTa](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLM-RoBERTa.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLM-RoBERTa.ipynb)
AlbertEmbeddings | [HuggingFace in Spark NLP - ALBERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20ALBERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20ALBERT.ipynb)
XlnetEmbeddings|[HuggingFace in Spark NLP - XLNet](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark_NLP%20-%20XLNet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark_NLP%20-%20XLNet.ipynb)

#### TF Hub to Spark NLP

Spark NLP | TF Hub Notebooks | Colab
:------------ | :-------------| :-------|
BertEmbeddings |  [TF Hub in Spark NLP - BERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20BERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20BERT.ipynb)
BertSentenceEmbeddings |  [TF Hub in Spark NLP - BERT Sentence](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb)
AlbertEmbeddings |  [TF Hub in Spark NLP - ALBERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20ALBERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20ALBERT.ipynb)

### Limitations

- If you are importing models from HuggingFace as Embeddings they must be for the `Fill-Mask` task. Meaning you cannot use a model in BertEmbeddings if they were trained or fine-tuned on token/text classification tasks in HuggingFace. They have a different architecture.
- There is a 2G size limitation with loading a TF SavedModel model in Spark NLP. Your model cannot be larger than 2G size or you will see the following error: `Required array size too large.` (We are working on going around this Java limitation, however, for the time being, there are some models which are over 2G and they are not compatible)
