---
layout: model
title: Persian BertForSequenceClassification Base Uncased model (from HooshvareLab)
author: John Snow Labs
name: bert_sequence_classifier_fa_base_uncased_sentiment_digikala
date: 2023-03-17
tags: [fa, open_source, bert, sequence_classification, ner, tensorflow]
task: Named Entity Recognition
language: fa
edition: Spark NLP 4.3.1
spark_version: 3.0
supported: true
engine: tensorflow
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bert-fa-base-uncased-sentiment-digikala` is a Persian model originally trained by `HooshvareLab`.

## Predicted Entities

`recommended`, `not_recommended`, `no_idea`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_fa_base_uncased_sentiment_digikala_fa_4.3.1_3.0_1679068309621.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_fa_base_uncased_sentiment_digikala_fa_4.3.1_3.0_1679068309621.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_fa_base_uncased_sentiment_digikala","fa") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

val sequenceClassifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_fa_base_uncased_sentiment_digikala","fa")
    .setInputCols(Array("document", "token"))
    .setOutputCol("ner")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_fa_base_uncased_sentiment_digikala|
|Compatibility:|Spark NLP 4.3.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|fa|
|Size:|609.3 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/HooshvareLab/bert-fa-base-uncased-sentiment-digikala
- https://github.com/hooshvare/parsbert
- https://www.digikala.com/opendata/
- https://www.digikala.com/opendata/
- https://colab.research.google.com/github/hooshvare/parsbert/blob/master/notebooks/Taaghche_Sentiment_Analysis.ipynb
- https://github.com/hooshvare/parsbert/issues