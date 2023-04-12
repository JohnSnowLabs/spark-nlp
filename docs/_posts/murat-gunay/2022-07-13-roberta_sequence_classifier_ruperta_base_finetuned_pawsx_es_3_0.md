---
layout: model
title: Spanish RobertaForSequenceClassification Base Cased model (from mrm8488)
author: John Snow Labs
name: roberta_sequence_classifier_ruperta_base_finetuned_pawsx
date: 2022-07-13
tags: [es, open_source, roberta, sequence_classification]
task: Text Classification
language: es
edition: Spark NLP 4.0.0
spark_version: 3.0
supported: true
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RobertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `RuPERTa-base-finetuned-pawsx-es` is a Spanish model originally trained by `mrm8488`.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_sequence_classifier_ruperta_base_finetuned_pawsx_es_4.0.0_3.0_1657716331993.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_sequence_classifier_ruperta_base_finetuned_pawsx_es_4.0.0_3.0_1657716331993.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

classifier = RoBertaForSequenceClassification.pretrained("roberta_sequence_classifier_ruperta_base_finetuned_pawsx","es") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, classifier])

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
 
val classifer = RoBertaForSequenceClassification.pretrained("roberta_sequence_classifier_ruperta_base_finetuned_pawsx","es") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, classifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_sequence_classifier_ruperta_base_finetuned_pawsx|
|Compatibility:|Spark NLP 4.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|472.9 MB|
|Case sensitive:|true|
|Max sentence length:|256|

## References

- https://huggingface.co/mrm8488/RuPERTa-base-finetuned-pawsx-es