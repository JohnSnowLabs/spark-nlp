---
layout: model
title: Turkish NERs with Bert
author: John Snow Labs
name: turkish_ner_bert
date: 2020-11-10
tags: [tr, open_source]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Named Entity Recognition (NER) deep learning model for Turkish texts. It recognizes Persons, Locations, and Organization entities using multi-lingual Bert word embedding. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER ç Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

## Predicted Entities

Persons (PER), Locations (LOC), Organizations (ORG)

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_TR/){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_TR.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/turkish_ner_bert_tr_2.6.2_2.4_1605043368882.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

Use as part of an NLP pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPython.html %}
```python
tr_ner = NerDLModel.pretrained(turkish_ner_bert, 'tr') \
               .setInputCols(["sentence", "token", "embeddings"]) \
               .setOutputCol("ner")
```
```scala
val tr_ner = NerDLModel.pretrained("turkish_ner_bert", "tr")
        .setInputCols(Array("sentence", "token", "embeddings"))
        .setOutputCol("ner")
```
</div>

## Results

```bash
+----------------------+---------+
|chunk                 |ner_label|
+----------------------+---------+
|William Henry Gates   |PER      |
|Microsoft             |ORG      |
|William Gates         |PER      |
|New Mexico            |LOC      |
|Albuquerque'de        |LOC      |
|Paul Allen            |PER      |
|B&Melinda G. Vakfı'nda|ORG      |
+----------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|turkish_ner_bert|
|Type:|ner|
|Compatibility:|Spark NLP 2.6.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|tr|
|Dependencies:|bert_multi_cased|

## Data Source

Trained on a custom dataset with multi-lingual Bert Embeddings (bert_multi_cased).

## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	         rec	         f1
B-LOC	 1949	 156	 158	 0.92589074	 0.9250119	 0.9254511
I-ORG	 1266	 266	 98	 0.8263708	 0.9281525	 0.8743094
I-LOC	 270	 54	 79	 0.8333333	 0.77363896	 0.8023774
I-PER	 1507	 89	 94	 0.94423556	 0.9412867	 0.94275886
B-ORG	 1805	 242	 119	 0.88177824	 0.9381497	 0.90909094
B-PER	 2841	 152	 267	 0.9492148	 0.91409266	 0.93132275
tp: 9638 fp: 959 fn: 815 labels: 6
Macro-average	 prec: 0.8934706, rec: 0.90338874, f1: 0.89840233
Micro-average	 prec: 0.9095027, rec: 0.92203194, f1: 0.91572446
```