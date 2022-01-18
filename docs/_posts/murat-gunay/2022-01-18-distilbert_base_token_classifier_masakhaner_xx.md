---
layout: model
title: NER Model for 9 African Languages
author: John Snow Labs
name: distilbert_base_token_classifier_masakhaner
date: 2022-01-18
tags: [xx, open_source]
task: Named Entity Recognition
language: xx
edition: Spark NLP 3.3.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is imported from `Hugging Face` ([link](https://huggingface.co/Davlan/distilbert-base-multilingual-cased-masakhaner)) and it's been finetuned on MasakhaNER dataset for 9 African languages (Hausa, Igbo, Kinyarwanda, Luganda, Nigerian, Pidgin, Swahilu, Wolof, and Yorùbá) leveraging `DistilBert` embeddings and `DistilBertForTokenClassification` for NER purposes.

## Predicted Entities

`DATE`, `LOC`, `ORG`, `PER`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/distilbert_base_token_classifier_masakhaner_xx_3.3.4_3.0_1642512428599.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetector = SentenceDetector()\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")

tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_base_token_classifier_masakhaner", "xx"))\
  .setInputCols(["sentence",'token'])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")
      
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])
text = """Adebiyi Adedayo jẹ ọkan ninu awọn oṣiṣẹ Naijiria ni Unilever FMCG, ọkan ninu awọn ile-iṣẹ Multinational ni Abuja Nigeria, lati Oṣu Kẹwa 1994."""
data = spark.createDataFrame([[text]]).toDF("text")

result = nlpPipeline.fit(data).transform(data)
```
```scala
val documentAssembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val sentenceDetector = SentenceDetector()
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

val tokenClassifier = DistilBertForTokenClassification.pretrained("distilbert_base_token_classifier_masakhaner", "xx"))\
  .setInputCols(Array("sentence","token"))
  .setOutputCol("ner")

val ner_converter = NerConverter()
      .setInputCols(Array("sentence", "token", "ner"))
      .setOutputCol("ner_chunk")
      
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Adebiyi Adedayo jẹ ọkan ninu awọn oṣiṣẹ Naijiria ni Unilever FMCG, ọkan ninu awọn ile-iṣẹ Multinational ni Abuja Nigeria, lati Oṣu Kẹwa 1994."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
+---------------+---------+
|chunk          |ner_label|
+---------------+---------+
|Adebiyi Adedayo|PER      |
|Naijiria       |LOC      |
|Unilever FMCG  |ORG      |
|Abuja Nigeria  |LOC      |
|Oṣu Kẹwa 1994  |DATE     |
+---------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|distilbert_base_token_classifier_masakhaner|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|xx|
|Size:|505.8 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## Data Source

[https://github.com/masakhane-io/masakhane-ner](https://github.com/masakhane-io/masakhane-ner)

## Benchmarking

```bash
language:   F1-score:
--------    --------
hau	     88.88
ibo	     84.87
kin	     74.19
lug	     78.43
luo	     73.32
pcm	     87.98
swa	     86.20
wol	     64.67
yor	     78.10
```