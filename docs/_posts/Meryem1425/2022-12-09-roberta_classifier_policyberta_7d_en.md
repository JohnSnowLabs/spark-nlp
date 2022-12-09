---
layout: model
title: English RobertaForSequenceClassification Cased model (from niksmer)
author: John Snow Labs
name: roberta_classifier_policyberta_7d
date: 2022-12-09
tags: [en, open_source, roberta, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.2.4
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `PolicyBERTa-7d` is a English model originally trained by `niksmer`.

## Predicted Entities

`social groups`, `economy`, `external relations`, `fabric of society`, `political system`, `welfare and quality of life`, `freedom and democracy`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_policyberta_7d_en_4.2.4_3.0_1670621489682.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_policyberta_7d","en") \
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
 
val roberta_classifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_policyberta_7d","en") 
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
|Model Name:|roberta_classifier_policyberta_7d|
|Compatibility:|Spark NLP 4.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|458.0 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

- https://huggingface.co/niksmer/PolicyBERTa-7d
- https://manifesto-project.wzb.eu/
- https://manifesto-project.wzb.eu/datasets
- https://manifesto-project.wzb.eu/down/papers/handbook_2021_version_5.pdf
- https://manifesto-project.wzb.eu/down/tutorials/main-dataset.html#measuring-parties-left-right-positions