---
layout: model
title: Danish XlmRobertaForSequenceClassification Cased model (from DaNLP)
author: John Snow Labs
name: xlmroberta_classifier_da_xlmr_ned
date: 2022-09-13
tags: [da, open_source, xlm_roberta, sequence_classification, classification]
task: Text Classification
language: da
edition: Spark NLP 4.1.0
spark_version: 3.0
supported: true
annotator: XlmRoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `da-xlmr-ned` is a Danish model originally trained by `DaNLP`.

## Predicted Entities

`not mentioned`, `mentioned`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_classifier_da_xlmr_ned_da_4.1.0_3.0_1663063060545.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

seq_classifier = XlmRoBertaForSequenceClassification.pretrained("xlmroberta_classifier_da_xlmr_ned","da") \
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
 
val seq_classifier = XlmRoBertaForSequenceClassification.pretrained("xlmroberta_classifier_da_xlmr_ned","da") 
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
|Model Name:|xlmroberta_classifier_da_xlmr_ned|
|Compatibility:|Spark NLP 4.1.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|da|
|Size:|882.3 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/DaNLP/da-xlmr-ned
- https://www.wikidata.org/wiki/Q182804
- https://danlp-alexandra.readthedocs.io/en/latest/docs/tasks/ned.html#xlmr
- https://danlp-alexandra.readthedocs.io/en/latest/docs/datasets.html#daned
- https://danlp-alexandra.readthedocs.io/en/latest/docs/datasets.html#dawikined