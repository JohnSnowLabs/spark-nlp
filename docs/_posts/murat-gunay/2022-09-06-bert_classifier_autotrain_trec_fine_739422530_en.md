---
layout: model
title: English BertForSequenceClassification Cased model (from ndavid)
author: John Snow Labs
name: bert_classifier_autotrain_trec_fine_739422530
date: 2022-09-06
tags: [en, open_source, bert, sequence_classification, classification]
task: Text Classification
language: en
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `autotrain-trec-fine-bert-739422530` is a English model originally trained by `ndavid`.

## Predicted Entities

`lang`, `cremat`, `speed`, `product`, `instru`, `religion`, `desc`, `manner`, `mount`, `dist`, `body`, `color`, `exp`, `def`, `sport`, `perc`, `dismed`, `other`, `ord`, `weight`, `count`, `food`, `currency`, `veh`, `volsize`, `techmeth`, `money`, `ind`, `word`, `substance`, `period`, `state`, `gr`, `code`, `plant`, `animal`, `date`, `symbol`, `title`, `abb`, `city`, `event`, `country`, `termeq`, `reason`, `temp`, `letter`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_autotrain_trec_fine_739422530_en_4.1.0_3.0_1662506819022.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_autotrain_trec_fine_739422530_en_4.1.0_3.0_1662506819022.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_autotrain_trec_fine_739422530","en") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, seq_classifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCols(Array("text")) 
      .setOutputCols(Array("document"))
      
val tokenizer = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_autotrain_trec_fine_739422530","en") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_autotrain_trec_fine_739422530|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.2 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/ndavid/autotrain-trec-fine-bert-739422530