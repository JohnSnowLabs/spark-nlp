---
layout: model
title: XLM-RoBERTa Base for Swahili (sent_xlm_roberta_base_finetuned_swahili)
author: John Snow Labs
name: sent_xlm_roberta_base_finetuned_swahili
date: 2021-10-15
tags: [open_source, xlm_roberta, embeddings, swahili, sw]
task: Embeddings
language: sw
edition: Spark NLP 3.3.1
spark_version: 3.0
supported: true
annotator: XlmRoBertaSentenceEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

**sent_xlm_roberta_base_finetuned_swahili** is a **Swahili RoBERTa** model obtained by fine-tuning **xlm-roberta-base** model on Swahili language texts. It provides **better performance** than the XLM-RoBERTa on named entity recognition datasets.

Specifically, this model is an *xlm-roberta-base* model that was fine-tuned on the **Swahili** corpus.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_xlm_roberta_base_finetuned_swahili_sw_3.3.1_3.0_1634304502531.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_xlm_roberta_base_finetuned_swahili_sw_3.3.1_3.0_1634304502531.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document = DocumentAssembler()\ 
.setInputCol("text")\ 
.setOutputCol("document")

sentencerDL = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\ 
.setInputCols(["document"])\ 
.setOutputCol("sentence")

sentece_embeddings = XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_base_finetuned_swahili", "sw")\ 
.setInputCols(["sentence"])\ 
.setOutputCol("sentence_embeddings")

```
```scala

val document = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
.setInputCols("document")
.setOutputCol("sentence")

val senteceEmbeddings = XlmRoBertaSentenceEmbeddings.pretrained("sent_xlm_roberta_base_finetuned_swahili", "sw")
.setInputCols("sentence")
.setOutputCol("sentence_embeddings")
```


{:.nlu-block}
```python
import nlu
nlu.load("sw.embed_sentence.xlm_roberta").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_xlm_roberta_base_finetuned_swahili|
|Compatibility:|Spark NLP 3.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence]|
|Output Labels:|[embeddings]|
|Language:|sw|
|Case sensitive:|true|

## Data Source

Model is trained by [David Adelani](https://huggingface.co/Davlan)

Improted from [https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili](https://huggingface.co/Davlan/xlm-roberta-base-finetuned-swahili)