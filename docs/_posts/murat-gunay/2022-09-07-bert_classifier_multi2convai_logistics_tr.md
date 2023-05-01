---
layout: model
title: Turkish BertForSequenceClassification Cased model (from inovex)
author: John Snow Labs
name: bert_classifier_multi2convai_logistics
date: 2022-09-07
tags: [tr, open_source, bert, sequence_classification, classification]
task: Text Classification
language: tr
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `multi2convai-logistics-tr-bert` is a Turkish model originally trained by `inovex`.

## Predicted Entities

`tour.job.delivered`, `tour.details`, `select`, `details.preferedNeighbour`, `tour.start`, `navigate`, `tour.job.signature`, `tour.job.carriedForward`, `details.address`, `details.avoidNeighbour`, `tour.postcode.select`, `help`, `safeplace`, `tour.job.safePlace`, `details.safeplace`, `undefined`, `tour.finish`, `yes`, `no`, `tour.job.failed`, `tour.job.collected`, `navigate.back`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_multi2convai_logistics_tr_4.1.0_3.0_1662514017718.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_multi2convai_logistics_tr_4.1.0_3.0_1662514017718.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_multi2convai_logistics","tr") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_multi2convai_logistics","tr") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")
   
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, seq_classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("tr.classify.bert.multi2convai.").predict("""PUT YOUR STRING HERE""")
```

</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_classifier_multi2convai_logistics|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|tr|
|Size:|415.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/inovex/multi2convai-logistics-tr-bert
- https://multi2convai/en/blog/use-cases
- https://github.com/inovex/multi2convai