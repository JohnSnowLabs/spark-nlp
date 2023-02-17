---
layout: docs
header: true
seotitle: Spark NLP
title: Transformers
permalink: /docs/en/transformers
key: docs-transformers
modify_date: "2022-09-27"
use_language_switcher: "Python-Scala-Java"
show_nav: true
sidebar:
    nav: sparknlp
---

<script> {% include scripts/transformerUseCaseSwitcher.js %} </script>

{% assign parent_path = "en/transformer_entries" %}

{% for file in site.static_files %}
    {% if file.path contains parent_path %}
        {% assign file_name = file.path | remove:  parent_path | remove:  "/" | prepend: "transformer_entries/" %}
        {% include_relative {{ file_name }} %}
    {% endif %}
{% endfor %}

## Import Transformers into Spark NLP

### Overview

We have extended support for `HuggingFace` 🤗   and `TF Hub` exported models since `3.1.0` to equivalent Spark NLP 🚀 annotators. Starting this release, you can easily use the `saved_model` feature in HuggingFace within a few lines of codes and import any `BERT`, `DistilBERT`, `CamemBERT`, `RoBERTa`, `DeBERTa`, `XLM-RoBERTa`, `Longformer`, `BertForTokenClassification`, `DistilBertForTokenClassification`, `AlbertForTokenClassification`, `RoBertaForTokenClassification`, `DeBertaForTokenClassification`, `XlmRoBertaForTokenClassification`, `XlnetForTokenClassification`,  `LongformerForTokenClassification`, `CamemBertForTokenClassification`, `CamemBertForSequenceClassification`, `BertForSequenceClassification`, `DistilBertForSequenceClassification`, `AlbertForSequenceClassification`, `RoBertaForSequenceClassification`, `DeBertaForSequenceClassification`, `XlmRoBertaForSequenceClassification`, `XlnetForSequenceClassification`,  `LongformerForSequenceClassification`, `AlbertForQuestionAnswering`, `BertForQuestionAnswering`,  `DeBertaForQuestionAnswering`, `DistilBertForQuestionAnswering`, `LongformerForQuestionAnswering`, `RoBertaForQuestionAnswering`, `XlmRoBertaForQuestionAnswering`, `TapasForQuestionAnswering`, and `Vision Transformers (ViT)` models to Spark NLP. We will work on the remaining annotators and extend this support to the rest with each release 😊

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
CamemBertEmbeddings|   |  ✅   | CamemBERT
RoBertaEmbeddings |   | ✅   | RoBERTa - DistilRoBERTa
DeBertaEmbeddings |   | ✅   | DeBERTa-v2 - DeBERTa-v3
XlmRoBertaEmbeddings |   | ✅   | XLM-RoBERTa
AlbertEmbeddings | ✅  |  ✅   |  ALBERT
XlnetEmbeddings |   | ✅  |  XLNet
LongformerEmbeddings |   | ✅  | Longformer
ElmoEmbeddings | ❎  |    |
UniversalSentenceEncoder |  ❎ |   |
BertForTokenClassification |   | ✅  |  [TFBertForTokenClassification](https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertForTokenClassification)
DistilBertForTokenClassification |   | ✅  |  [TFDistilBertForTokenClassification](https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.TFDistilBertForTokenClassification)
AlbertForTokenClassification |   | ✅  |  [TFAlbertForTokenClassification](https://huggingface.co/docs/transformers/model_doc/albert#transformers.TFAlbertForTokenClassification)
RoBertaForTokenClassification |   | ✅  |  [TFRobertaForTokenClassification](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.TFRobertaForTokenClassification)
DeBertaForTokenClassification |   | ✅  |  [TFDebertaV2ForTokenClassification](https://huggingface.co/docs/transformers/model_doc/deberta-v2#transformers.TFDebertaV2ForTokenClassification)
XlmRoBertaForTokenClassification |   | ✅  |  [TFXLMRobertaForTokenClassification](https://huggingface.co/docs/transformers/model_doc/xlmroberta#transformers.TFXLMRobertaForTokenClassification)
XlnetForTokenClassification |   | ✅  |  [TFXLNetForTokenClassificationet](https://huggingface.co/docs/transformers/model_doc/xlnet#transformers.TFXLNetForTokenClassificationet)
LongformerForTokenClassification |   | ✅  |  [TFLongformerForTokenClassification](https://huggingface.co/docs/transformers/model_doc/longformer#transformers.TFLongformerForTokenClassification)
CamemBertForTokenClassification |   | ✅  |  [TFCamemBertForTokenClassification](https://huggingface.co/docs/transformers/model_doc/camembert#transformers.TFCamembertForTokenClassification)
CamemBertForSequenceClassification |   | ✅  |  [TFCamemBertForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/camembert#transformers.TFCamembertForSequenceClassification)
BertForSequenceClassification |   | ✅  |  [TFBertForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertForSequenceClassification)
DistilBertForSequenceClassification |   | ✅  |  [TFDistilBertForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.TFDistilBertForSequenceClassification)
AlbertForSequenceClassification |   | ✅  |  [TFAlbertForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/albert#transformers.TFAlbertForSequenceClassification)
RoBertaForSequenceClassification |   | ✅  |  [TFRobertaForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.TFRobertaForSequenceClassification)
DeBertaForSequenceClassification |   | ✅  |  [TFDebertaV2ForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/deberta-v2#transformers.TFDebertaV2ForSequenceClassification)
XlmRoBertaForSequenceClassification |   | ✅  |  [TFXLMRobertaForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.TFXLMRobertaForSequenceClassification)
XlnetForSequenceClassification |   | ✅  |  [TFXLNetForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/xlnet#transformers.TFXLNetForSequenceClassification)
LongformerForSequenceClassification |   | ✅  |  [TFLongformerForSequenceClassification](https://huggingface.co/docs/transformers/model_doc/longformer#transformers.TFLongformerForSequenceClassification)
| AlbertForQuestionAnswering  |   |✅  |  [TFAlbertForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/albert#transformers.TFAlbertForQuestionAnswering)
| BertForQuestionAnswering  |   |✅  |  [TFBertForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/bert#transformers.TFBertForQuestionAnswering)
| DeBertaForQuestionAnswering |   |✅  |  [TFDebertaV2ForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/deberta-v2#transformers.TFDebertaV2ForQuestionAnswering)
| DistilBertForQuestionAnswering  |   |✅  |  [TFDistilBertForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.TFDistilBertForQuestionAnswering)
| LongformerForQuestionAnswering  |   |✅  |  [TFLongformerForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/longformer#transformers.TFLongformerForQuestionAnswering)
| RoBertaForQuestionAnswering  |   |✅  |  [TFRobertaForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/roberta#transformers.TFRobertaForQuestionAnswering)
| XlmRoBertaForQuestionAnswering  |   |✅  |  [TFXLMRobertaForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/xlm-roberta#transformers.TFXLMRobertaForQuestionAnswering)
| TapasForQuestionAnswering  |   | ❎ |  [TFTapasForQuestionAnswering](https://huggingface.co/docs/transformers/model_doc/tapas#transformers.TFTapasForQuestionAnswering)
| ViTForImageClassification | ❌  | ✅ | [TFViTForImageClassification](https://huggingface.co/docs/transformers/model_doc/vit#transformers.TFViTForImageClassification)
| Automatic Speech Recognition (Wav2Vec2ForCTC)|   | ❎ | [TFWav2Vec2ForCTC](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.TFWav2Vec2ForCTC)
| T5Transformer |   |  ❌ |
| MarianTransformer|   | ❌  |
| OpenAI GPT2|   | ❌  |

### Example Notebooks

#### HuggingFace to Spark NLP

Spark NLP | HuggingFace Notebooks | Colab
:------------ | :-------------| :----------|
BertEmbeddings |  [HuggingFace in Spark NLP - BERT](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT.ipynb)
BertSentenceEmbeddings | [HuggingFace in Spark NLP - BERT Sentence](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb)
DistilBertEmbeddings| [HuggingFace in Spark NLP - DistilBERT](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBERT.ipynb)
CamemBertEmbeddings| [HuggingFace in Spark NLP - CamemBERT](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20CamemBERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20CamemBERT.ipynb)
RoBertaEmbeddings | [HuggingFace in Spark NLP - RoBERTa](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBERTa.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBERTa.ipynb)
DeBertaEmbeddings | [HuggingFace in Spark NLP - DeBERTa](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DeBERTa.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DeBERTa.ipynb)
XlmRoBertaEmbeddings | [HuggingFace in Spark NLP - XLM-RoBERTa](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLM-RoBERTa.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLM-RoBERTa.ipynb)
AlbertEmbeddings | [HuggingFace in Spark NLP - ALBERT](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20ALBERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20ALBERT.ipynb)
XlnetEmbeddings|[HuggingFace in Spark NLP - XLNet](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLNet.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XLNet.ipynb)
LongformerEmbeddings|[HuggingFace in Spark NLP - Longformer](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20Longformer.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20Longformer.ipynb)
BertForTokenClassification|[HuggingFace in Spark NLP - BertForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForTokenClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForTokenClassification.ipynb)
DistilBertForTokenClassification|[HuggingFace in Spark NLP - DistilBertForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBertForTokenClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBertForTokenClassification.ipynb)
AlbertForTokenClassification|[HuggingFace in Spark NLP - AlbertForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20AlbertForTokenClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20AlbertForTokenClassification.ipynb)
RoBertaForTokenClassification|[HuggingFace in Spark NLP - RoBertaForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBertaForTokenClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBertaForTokenClassification.ipynb)
XlmRoBertaForTokenClassification|[HuggingFace in Spark NLP - XlmRoBertaForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XlmRoBertaForTokenClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XlmRoBertaForTokenClassification.ipynb)
CamemBertForTokenClassification|[HuggingFace in Spark NLP - CamemBertForTokenClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20CamemBertForTokenClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20CamemBertForTokenClassification.ipynb)
CamemBertForTokenClassification|[HuggingFace in Spark NLP - CamemBertForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20CamemBertForSequenceClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20CamemBertForSequenceClassification.ipynb)
BertForSequenceClassification |[HuggingFace in Spark NLP - BertForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForSequenceClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForSequenceClassification.ipynb)
DistilBertForSequenceClassification |[HuggingFace in Spark NLP - DistilBertForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBertForSequenceClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBertForSequenceClassification.ipynb)
AlbertForSequenceClassification |[HuggingFace in Spark NLP - AlbertForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForSequenceClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20AlbertForSequenceClassification.ipynb)
RoBertaForSequenceClassification |[HuggingFace in Spark NLP - RoBertaForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForSequenceClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBertaForSequenceClassification.ipynb)
XlmRoBertaForSequenceClassification |[HuggingFace in Spark NLP - XlmRoBertaForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForSequenceClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XlmRoBertaForSequenceClassification.ipynb)
XlnetForSequenceClassification |[HuggingFace in Spark NLP - XlnetForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForSequenceClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XlnetForSequenceClassification.ipynb)
LongformerForSequenceClassification |[HuggingFace in Spark NLP - LongformerForSequenceClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForSequenceClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20LongformerForSequenceClassification.ipynb)
AlbertForQuestionAnswering |[HuggingFace in Spark NLP - AlbertForQuestionAnswering](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20AlbertForQuestionAnswering.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20AlbertForQuestionAnswering.ipynb)
BertForQuestionAnswering|[HuggingFace in Spark NLP - BertForQuestionAnswering](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForQuestionAnswering.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20BertForQuestionAnswering.ipynb)
DeBertaForQuestionAnswering|[HuggingFace in Spark NLP - DeBertaForQuestionAnswering](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DeBertaForQuestionAnswering.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DeBertaForQuestionAnswering.ipynb)
DistilBertForQuestionAnswering|[HuggingFace in Spark NLP - DistilBertForQuestionAnswering](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBertForQuestionAnswering.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20DistilBertForQuestionAnswering.ipynb)
LongformerForQuestionAnswering|[HuggingFace in Spark NLP - LongformerForQuestionAnswering](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20LongformerForQuestionAnswering.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20LongformerForQuestionAnswering.ipynb)
RoBertaForQuestionAnswering|[HuggingFace in Spark NLP - RoBertaForQuestionAnswering](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBertaForQuestionAnswering.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20RoBertaForQuestionAnswering.ipynb)
XlmRobertaForQuestionAnswering|[HuggingFace in Spark NLP - XlmRobertaForQuestionAnswering](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XlmRobertaForQuestionAnswering.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20XlmRobertaForQuestionAnswering.ipynb)
ViTForImageClassification|[HuggingFace in Spark NLP - ViTForImageClassification](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/HuggingFace%20in%20Spark%20NLP%20-%20ViTForImageClassification.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/HuggingFace%20in%20Spark%20NLP%20-%20ViTForImageClassification.ipynb)


#### TF Hub to Spark NLP

Spark NLP | TF Hub Notebooks | Colab
:------------ | :-------------| :-------|
BertEmbeddings |  [TF Hub in Spark NLP - BERT](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20BERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20BERT.ipynb)
BertSentenceEmbeddings |  [TF Hub in Spark NLP - BERT Sentence](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20BERT%20Sentence.ipynb)
AlbertEmbeddings |  [TF Hub in Spark NLP - ALBERT](https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20ALBERT.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/transformers/TF%20Hub%20in%20Spark%20NLP%20-%20ALBERT.ipynb)
