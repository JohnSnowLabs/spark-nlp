---
layout: model
title: Sentiment Analysis in Spanish
author: John Snow Labs
name: beto_sentiment_analysis
date: 2022-07-17
tags: [beto, sentiment, bertforsequenceclassification, es, open_source]
task: Text Classification
language: es
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. beto_sentiment_analysis is a Spanish model orginally trained with TASS 2020 corpus (around ~5k tweets) of several dialects of Spanish.

## Predicted Entities

`POS`, `NEG`, `NEU`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/beto_sentiment_analysis_es_4.0.0_3.0_1658072197787.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

sequenceClassifier = BertForSequenceClassification.pretrained("beto_sentiment_analysis", "es")\
  .setInputCols(["document",'token'])\
  .setOutputCol("class")

pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    sequenceClassifier    
])

# couple of simple examples
example = spark.createDataFrame([["Te quiero. Te amo."]]).toDF("text")

result = pipeline.fit(example).transform(example)

# result is a DataFrame
result.select("text", "class.result").show()
```
```scala
val document_assembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val tokenizer = new Tokenizer() 
    .setInputCols(Array("document")) 
    .setOutputCol("token")

val sequenceClassifier = BertForSequenceClassification.pretrained("beto_sentiment_analysis", "es")
  .setInputCols(Array("document","token"))
  .setOutputCol("class")

val pipeline = new Pipeline.setStages(Array(document_assembler, tokenizer, sequenceClassifier))

# couple of simple examples
val example = Seq("Te quiero. Te amo.").toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
+------------------+------+
|              text|result|
+------------------+------+
|Te quiero. Te amo.| [POS]|
+------------------+------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|beto_sentiment_analysis|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|412.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

https://github.com/pysentimiento/pysentimiento/