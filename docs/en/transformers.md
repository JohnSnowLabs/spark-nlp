---
layout: docs
header: true
title: Transformers
permalink: /docs/en/transformers
key: docs-transformers
modify_date: "2019-11-01"
use_language_switcher: "Python-Scala-Java"
---

<div class="h3-box" markdown="1">

These are the [Transformers](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)) that are available in Spark NLP. The created embeddings can be combined with
DL based [Annotators](https://nlp.johnsnowlabs.com/docs/en/annotators), such as the `ClassifierDL` and `NerDL` to achieve state-of-the-art results in a wide range of language tasks.

## Import Transformers into Spark NLP

Starting with Spark NLP 3.1.0 the support for HuggingFace exported models in equivalent Spark NLP annotators has been extended. Users can now easily use `saved_model` feature in HuggingFace within a few lines of codes and import any BERT, DistilBERT, RoBERTa, and XLM-RoBERTa models to Spark NLP. We will work on the remaining annotators and extend this support to the rest with each release.

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
BertEmbeddings |  ❎  |  ✅  |  BERT - Small BERT - ELECTRA
BertSentenceEmbeddings |  ❎  | ✅   | BERT - Small BERT - ELECTRA
DistilBertEmbeddings|   |  ✅   | DistilBERT
RoBertaEmbeddings |   | ✅   | RoBERTa - DistilRoBERTa
XlmRoBertaEmbeddings |   | ✅   | XLM-RoBERTa
AlbertEmbeddings | ❎  |  ❎  |
XlnetEmbeddings |   | ❎  |
ElmoEmbeddings | ❎  |  ❎  |
UniversalSentenceEncoder |  ❎ |   |
T5Transformer |   |  ❌ |
MarianTransformer|   | ❌  |

### Notebooks

Spark NLP | HuggingFace Notebooks
:------------ | :-------------|
BertEmbeddings |  [HuggingFace in Spark NLP - BERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT.ipynb)
BertSentenceEmbeddings | [HuggingFace in Spark NLP - BERT Sentence](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb)
DistilBertEmbeddings| [HuggingFace in Spark NLP - DistilBERT](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBERT.ipynb)
RoBertaEmbeddings | [HuggingFace in Spark NLP - RoBERTa](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBERTa.ipynb)
XlmRoBertaEmbeddings | [HuggingFace in Spark NLP - XLM-RoBERTa](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/jupyter/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLM-RoBERTa.ipynb)


### Limitations

- If you are importing models from HuggingFace as Embeddings they must be for `Fill-Mask` task. Meaning you cannot use a model in BertEmbeddings if they were trained or fine-tuned on token/text classification tasks in HuggingFace. They have a different architecture.
- There is a 2G size limitation with loading a TF SavedModel model in Spark NLP. Your model cannot be larger than 2G size or you will see the following error: `Required array size too large. (We are working on going around this Java limitation, however, for the time being, there are some models which are over 2G and they are not compatible)`

</div>

<script> {% include scripts/approachModelSwitcher.js %} </script>

{% assign parent_path = "en/transformer_entries" %}

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "transformer_entries/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}