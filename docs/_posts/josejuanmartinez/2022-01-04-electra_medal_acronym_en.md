---
layout: model
title: Electra MeDAL Acronym BERT Embeddings
author: John Snow Labs
name: electra_medal_acronym
date: 2022-01-04
tags: [acronym, abbreviation, en, open_source]
task: Embeddings
language: en
edition: Spark NLP 3.3.3
spark_version: 3.0
supported: true
annotator: BertEmbeddings
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Electra model fine tuned on MeDAL, a large dataset on abbreviation disambiguation, designed for pretraining natural language understanding models in the medical domain. Check the reference [here](https://aclanthology.org/2020.clinicalnlp-1.15.pdf).

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/electra_medal_acronym_en_3.3.3_3.0_1641310227830.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler= DocumentAssembler()\
.setInputCol("text")\
.setOutputCol("document")

sentenceDetector = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer= Tokenizer()\
.setInputCols(["sentence"])\
.setOutputCol("token")

embeddings = BertEmbeddings.pretrained("electra_medal_acronym", "en") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

nlpPipeline= Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, embeddings])
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentenceDetector = new SentenceDetector()
.setInputCols(Array("document"))
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols(Array("sentence"))
.setOutputCol("token")

val embeddings = BertEmbeddings.pretrained("electra_medal_acronym", "en") 
.setInputCols("sentence", "token")
.setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, embeddings))
```


{:.nlu-block}
```python
import nlu
nlu.load("en.embed.electra.medical").predict("""Put your text here.""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|electra_medal_acronym|
|Compatibility:|Spark NLP 3.3.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[electra]|
|Language:|en|
|Size:|66.0 MB|
|Case sensitive:|true|

## Data Source

https://github.com/BruceWen120/medal
