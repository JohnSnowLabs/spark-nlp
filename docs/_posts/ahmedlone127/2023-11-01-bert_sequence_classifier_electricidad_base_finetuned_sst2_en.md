---
layout: model
title: Spanish BertForSequenceClassification Base Cased model (from mrm8488)
author: John Snow Labs
name: bert_sequence_classifier_electricidad_base_finetuned_sst2
date: 2023-11-01
tags: [bert, es, sequence_classification, open_source, en, onnx]
task: Text Classification
language: en
edition: Spark NLP 5.1.4
spark_version: 3.4
supported: true
engine: onnx
annotator: BertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP. `electricidad-base-finetuned-sst2-es` is a Spanish model originally trained by `mrm8488`.

## Predicted Entities

`NEG`, `NEU`, `POS`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_electricidad_base_finetuned_sst2_en_5.1.4_3.4_1698808012843.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_sequence_classifier_electricidad_base_finetuned_sst2_en_5.1.4_3.4_1698808012843.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
  
classifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_electricidad_base_finetuned_sst2","es")     .setInputCols(["document", "token"])     .setOutputCol("class")
    
pipeline = Pipeline(stages=[documentAssembler, tokenizer, classifier])

data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val documentAssembler = new DocumentAssembler() 
      .setInputCol("text") 
      .setOutputCol("document")
 
val tokenizer = new Tokenizer() 
    .setInputCols(Array("document"))
    .setOutputCol("token")

val classifier = BertForSequenceClassification.pretrained("bert_sequence_classifier_electricidad_base_finetuned_sst2","es") 
    .setInputCols(Array("document", "token")) 
    .setOutputCol("embeddings")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, classifier))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
nlu.load("en.classify.bert.base_finetuned").predict("""I love Spark NLP""")
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_sequence_classifier_electricidad_base_finetuned_sst2|
|Compatibility:|Spark NLP 5.1.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|410.5 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

References

https://huggingface.co/mrm8488/electricidad-base-finetuned-sst2-es