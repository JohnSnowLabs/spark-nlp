---
layout: model
title: BERT Sequence Classification - Identify Trec Data Classes
author: John Snow Labs
name: bert_sequence_classifier_trec_coarse
date: 2021-11-06
tags: [bert_for_sequence_classification, trec, en, open_source]
task: Text Classification
language: en
edition: Spark NLP 3.3.2
spark_version: 2.4
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is imported from `Hugging Face-models` and it is a simple base BERT model trained on the "trec" dataset.

## Predicted Entities

`DESC`, `ENTY`, `HUM`, `NUM`, `ABBR`, `LOC`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_trec_coarse_en_3.3.2_2.4_1636229841055.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_trec_coarse_en_3.3.2_2.4_1636229841055.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier = BertForSequenceClassification \
      .pretrained('bert_sequence_classifier_trec_coarse', 'en') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([['Germany is the largest country in Europe economically.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = Tokenizer() 
    .setInputCols("document") 
    .setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_trec_coarse", "en")
      .setInputCols("document", "token")
      .setOutputCol("class")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq.empty["Germany is the largest country in Europe economically."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
['LOC']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_trec_coarse|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[label]|
|Language:|en|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/aychang/bert-base-cased-trec-coarse](https://huggingface.co/aychang/bert-base-cased-trec-coarse)

## Benchmarking

```bash
epoch: 2.0, eval_loss: 0.138086199760437
eval_runtime: 1.6132, eval_samples_per_second: 309.94

+------------+-------+-----------------+--------------+
|      entity|eval_f1|   eval_precision|   eval_recall|
+------------+-------+-----------------+--------------+
|        DESC|  0.981|            0.985|         0.978|
|        ENTY|  0.944|            0.988|         0.904| 
|        ABBR|     1.|               1.|            1.| 
|         HUM|  0.992|            0.984|            1.|
|         NUM|  0.969|            0.941|            1.|
|         LOC|  0.981|            0.975|         0.987|
+------------+-------+-----------------+--------------+

eval_accuracy:  0.974
```
