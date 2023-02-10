---
layout: model
title: Dutch BertForSequenceClassification Cased model (from RuudVelo)
author: John Snow Labs
name: bert_classifier_dutch_news_clf_finetuned
date: 2022-09-19
tags: [bert, sequence_classification, classification, open_source, nl]
task: Text Classification
language: nl
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `dutch_news_clf_bert_finetuned` is a Dutch model originally trained by `RuudVelo`.

## Predicted Entities

`Economie`, `Buitenland`, `Politiek`, `Regionaal nieuws`, `Tech`, `Koningshuis`, `Binnenland`, `Cultuur & Media`, `Opmerkelijk`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_dutch_news_clf_finetuned_nl_4.1.0_3.0_1663608270923.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_dutch_news_clf_finetuned_nl_4.1.0_3.0_1663608270923.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_dutch_news_clf_finetuned","nl") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")

pipeline = Pipeline(stages=[documentAssembler, tokenizer,sequenceClassifier_loaded])

data = spark.createDataFrame([["Ik hou van Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
          .setInputCol("text") 
          .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier_loaded = BertForSequenceClassification.pretrained("bert_classifier_dutch_news_clf_finetuned","nl") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer,sequenceClassifier_loaded))

val data = Seq("Ik hou van Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_dutch_news_clf_finetuned|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|nl|
|Size:|409.6 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/RuudVelo/dutch_news_clf_bert_finetuned