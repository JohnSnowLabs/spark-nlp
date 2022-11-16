---
layout: model
title: Part of Speech for Indonesian
author: John Snow Labs
name: roberta_token_classifier_pos_tagger
date: 2021-12-27
tags: [indonesian, roberta, pos, id, open_source]
task: Part of Speech Tagging
language: id
edition: Spark NLP 3.3.4
spark_version: 2.4
supported: true
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from `Hugging Face` and it's been fine-tuned on [indonlu's](https://hf.co/datasets/indonlu) POSP dataset for the Indonesian language, leveraging `RoBERTa` embeddings and `RobertaForTokenClassification` for POS tagging purposes.

## Predicted Entities

`PPO`, `KUA`, `ADV`, `PRN`, `VBI`, `PAR`, `VBP`, `NNP`, `UNS`, `VBT`, `VBL`, `NNO`, `ADJ`, `PRR`, `PRK`, `CCN`, `$$$`, `ADK`, `ART`, `CSN`, `NUM`, `SYM`, `INT`, `NEG`, `PRI`, `VBE`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_token_classifier_pos_tagger_id_3.3.4_2.4_1640589883082.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_token_classifier_pos_tagger", "id"))\
  .setInputCols(["sentence",'token'])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")
      
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """Budi sedang pergi ke pasar."""
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

val tokenClassifier = RoBertaForTokenClassification.pretrained("roberta_token_classifier_pos_tagger", "id"))\
  .setInputCols(Array("sentence","token"))\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(Array("sentence", "token", "ner"))\
      .setOutputCol("ner_chunk")
      
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["Budi sedang pergi ke pasar."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
+------+---------+
|chunk |ner_label|
+------+---------+
|Budi  |NNO      |
|sedang|ADK      |
|pergi |VBI      |
|ke    |PPO      |
|pasar |NNO      |
|.     |SYM      |
+------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_token_classifier_pos_tagger|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|id|
|Size:|466.2 MB|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

[https://huggingface.co/w11wo/indonesian-roberta-base-posp-tagger](https://huggingface.co/w11wo/indonesian-roberta-base-posp-tagger)

## Benchmarking

```bash
   label      score
      f1     0.8893
Accuracy     0.9399
```