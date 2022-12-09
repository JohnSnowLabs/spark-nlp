---
layout: model
title: Spanish RobertaForSequenceClassification Base Cased model (from edumunozsala)
author: John Snow Labs
name: roberta_classifier_bertin_base_sentiment_analysis
date: 2022-12-09
tags: [es, open_source, roberta, sequence_classification, classification, tensorflow]
task: Text Classification
language: es
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
engine: tensorflow
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `bertin_base_sentiment_analysis_es` is a Spanish model originally trained by `edumunozsala`.

## Predicted Entities

`Negativo`, `Positivo`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_bertin_base_sentiment_analysis_es_4.2.4_3.0_1670624545476.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
    .setInputCols(["text"]) \
    .setOutputCols("document")

tokenizer = Tokenizer() \
    .setInputCols("document") \
    .setOutputCol("token")

roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_bertin_base_sentiment_analysis","es") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, roberta_classifier])

data = spark.createDataFrame([["I love you!"], ["I feel lucky to be here."]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
    .setInputCols("text")
    .setOutputCols("document")
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_bertin_base_sentiment_analysis","es") 
    .setInputCols(Array("document", "token"))
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, roberta_classifier))

val data = Seq("I love you!").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_bertin_base_sentiment_analysis|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|455.6 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/edumunozsala/bertin_base_sentiment_analysis_es
- http://journal.sepln.org/sepln/ojs/ojs/index.php/pln/article/view/6403
- https://github.com/edumunozsala
- https://paperswithcode.com/sota?task=Sentiment+Analysis&dataset=IMDb+Reviews+in+Spanish