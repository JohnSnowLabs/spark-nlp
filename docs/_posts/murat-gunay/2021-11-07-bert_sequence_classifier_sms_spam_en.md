---
layout: model
title: BERT Sequence Classification - Detect Spam SMS
author: John Snow Labs
name: bert_sequence_classifier_sms_spam
date: 2021-11-07
tags: [sms, spam, bert_for_sequence_classification, en, open_source]
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

This model is imported from `Hugging Face-models`. It is a BERT-Tiny version of the `sms_spam` dataset. It identifies if the SMS is spam or not.

- `LABEL_0` : No Spam
- `LABEL_1` : Spam

## Predicted Entities

`LABEL_0`, `LABEL_1`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_sms_spam_en_3.3.2_2.4_1636290194115.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_sms_spam_en_3.3.2_2.4_1636290194115.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
      .pretrained('bert_sequence_classifier_sms_spam', 'en') \
      .setInputCols(['token', 'document']) \
      .setOutputCol('class') \
      .setCaseSensitive(True) \
      .setMaxSentenceLength(512)

pipeline = Pipeline(stages=[document_assembler, tokenizer, sequenceClassifier])

example = spark.createDataFrame([['Camera - You are awarded a SiPix Digital Camera! call 09061221066 from landline. Delivery within 28 days.']]).toDF("text")
result = pipeline.fit(example).transform(example)
```
```scala
val document_assembler = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val tokenClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_sms_spam", "en")
      .setInputCols("document", "token")
      .setOutputCol("class")
      .setCaseSensitive(true)
      .setMaxSentenceLength(512)

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, sequenceClassifier))

val example = Seq.empty["Camera - You are awarded a SiPix Digital Camera! call 09061221066 from landline. Delivery within 28 days."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
['LABEL_1']
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_sms_spam|
|Compatibility:|Spark NLP 3.3.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[token, sentence]|
|Output Labels:|[label]|
|Language:|en|
|Case sensitive:|true|

## Data Source

[https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection](https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection)

## Benchmarking

```bash
   label  score
accuracy   0.98
```
