---
layout: model
title: Danish BertForSequenceClassification Cased model (from RJuro)
author: John Snow Labs
name: bert_classifier_da_hyggebert
date: 2022-09-06
tags: [da, open_source, bert, sequence_classification, classification]
task: Text Classification
language: da
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `Da-HyggeBERT` is a Danish model originally trained by `RJuro`.

## Predicted Entities

`begær`, `stolthed`, `misbilligelse`, `afsky`, `irritation`, `fornøjelse`, `indsigt`, `overraskelse`, `frygt`, `nysgerrighed`, `tristhed`, `taknemmelighed`, `skuffelse`, `optimisme`, `nervøsitet`, `beundring`, `kærlighed`, `vrede`, `spænding`, `lettelse`, `forlegenhed`, `medhold`, `glæde`, `neutral`, `sorg`, `omsorg`, `fortrydelse`, `forvirring`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_classifier_da_hyggebert_da_4.1.0_3.0_1662499952387.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_classifier_da_hyggebert_da_4.1.0_3.0_1662499952387.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_da_hyggebert","da") \
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
 
val seq_classifier = BertForSequenceClassification.pretrained("bert_classifier_da_hyggebert","da") 
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
|Model Name:|bert_classifier_da_hyggebert|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|da|
|Size:|415.2 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/RJuro/Da-HyggeBERT