---
layout: model
title: Castilian, Spanish roberta_classifier_bertin_base_spanish_semitic_languages_eval_2018_task_1 RoBertaForSequenceClassification from maxpe
author: John Snow Labs
name: roberta_classifier_bertin_base_spanish_semitic_languages_eval_2018_task_1
date: 2023-11-29
tags: [roberta, es, open_source, sequence_classification, onnx]
task: Text Classification
language: es
edition: Spark NLP 5.2.0
spark_version: 3.0
supported: true
engine: onnx
annotator: RoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_classifier_bertin_base_spanish_semitic_languages_eval_2018_task_1` is a Castilian, Spanish model originally trained by maxpe.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_classifier_bertin_base_spanish_semitic_languages_eval_2018_task_1_es_5.2.0_3.0_1701222594964.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_classifier_bertin_base_spanish_semitic_languages_eval_2018_task_1_es_5.2.0_3.0_1701222594964.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = Tokenizer()\
    .setInputCols("document")\
    .setOutputCol("token")  
    
sequenceClassifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_bertin_base_spanish_semitic_languages_eval_2018_task_1","es")\
            .setInputCols(["document","token"])\
            .setOutputCol("class")

pipeline = Pipeline().setStages([document_assembler, tokenizer, sequenceClassifier])

data = spark.createDataFrame([["PUT YOUR STRING HERE"]]).toDF("text")

result = pipeline.fit(data).transform(data)

```
```scala

val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val tokenizer = new Tokenizer()
    .setInputCols("document") 
    .setOutputCol("token")  
    
val sequenceClassifier = RoBertaForSequenceClassification.pretrained("roberta_classifier_bertin_base_spanish_semitic_languages_eval_2018_task_1","es")
            .setInputCols(Array("document","token"))
            .setOutputCol("class")

val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))

val data = Seq("PUT YOUR STRING HERE").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)


```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_classifier_bertin_base_spanish_semitic_languages_eval_2018_task_1|
|Compatibility:|Spark NLP 5.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|es|
|Size:|464.5 MB|

## References

https://huggingface.co/maxpe/bertin-roberta-base-spanish_sem_eval_2018_task_1