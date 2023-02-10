---
layout: model
title: Dutch NER Model
author: John Snow Labs
name: bert_token_classifier_dutch_udlassy_ner
date: 2021-12-08
tags: [dutch, token_classifier, bert, ner, nl, open_source]
task: Named Entity Recognition
language: nl
edition: Spark NLP 3.3.2
spark_version: 2.4
supported: true
annotator: BertForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face`, and has been fine-tuned on Universal Dependencies Lassy dataset for Dutch language, leveraging `Bert` embeddings and `BertForTokenClassification` for NER purposes.

## Predicted Entities

`CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_dutch_udlassy_ner_nl_3.3.2_2.4_1638958339134.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_dutch_udlassy_ner_nl_3.3.2_2.4_1638958339134.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_dutch_udlassy_ner", "nl"))\
  .setInputCols(["sentence",'token'])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")
      
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """Mijn naam is Peter Fergusson. Ik woon sinds oktober 2011 in New York en werk 5 jaar bij Tesla Motor."""
result = model.transform(spark.createDataFrame([[text]]).toDF("text"))
```
```scala
val documentAssembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

val tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_dutch_udlassy_ner", "nl"))\
  .setInputCols(Array("sentence","token"))\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(Array("sentence", "token", "ner"))\
      .setOutputCol("ner_chunk")
      
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Mijn naam is Peter Fergusson. Ik woon sinds oktober 2011 in New York en werk 5 jaar bij Tesla Motor."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```


{:.nlu-block}
```python
import nlu
nlu.load("nl.ner.bert").predict("""Mijn naam is Peter Fergusson. Ik woon sinds oktober 2011 in New York en werk 5 jaar bij Tesla Motor.""")
```

</div>

## Results

```bash
+------------------------+---------+
|chunk                   |ner_label|
+------------------------+---------+
|Peter Fergusson         |PERSON   |
|oktober 2011            |DATE     |
|New York                |GPE      |
|5 jaar                  |DATE     |
|Tesla Motor             |ORG      |
+------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_dutch_udlassy_ner|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|nl|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

[https://huggingface.co/wietsedv/bert-base-dutch-cased-finetuned-udlassy-ner](https://huggingface.co/wietsedv/bert-base-dutch-cased-finetuned-udlassy-ner)
