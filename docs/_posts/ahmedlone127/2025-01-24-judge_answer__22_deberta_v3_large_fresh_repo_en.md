---
layout: model
title: English judge_answer__22_deberta_v3_large_fresh_repo DeBertaForSequenceClassification from tom-010
author: John Snow Labs
name: judge_answer__22_deberta_v3_large_fresh_repo
date: 2025-01-24
tags: [en, open_source, onnx, sequence_classification, deberta]
task: Text Classification
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: DeBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`judge_answer__22_deberta_v3_large_fresh_repo` is a English model originally trained by tom-010.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/judge_answer__22_deberta_v3_large_fresh_repo_en_5.5.1_3.0_1737728167299.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/judge_answer__22_deberta_v3_large_fresh_repo_en_5.5.1_3.0_1737728167299.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
     
documentAssembler = DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')
    
tokenizer = Tokenizer() \
    .setInputCols(['document']) \
    .setOutputCol('token')

sequenceClassifier  = DeBertaForSequenceClassification.pretrained("judge_answer__22_deberta_v3_large_fresh_repo","en") \
     .setInputCols(["documents","token"]) \
     .setOutputCol("class")

pipeline = Pipeline().setStages([documentAssembler, tokenizer, sequenceClassifier])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler()
    .setInputCols("text")
    .setOutputCols("document")
    
val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

val sequenceClassifier = DeBertaForSequenceClassification.pretrained("judge_answer__22_deberta_v3_large_fresh_repo", "en")
    .setInputCols(Array("documents","token")) 
    .setOutputCol("class") 
    
val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, sequenceClassifier))
val data = Seq("I love spark-nlp").toDS.toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|judge_answer__22_deberta_v3_large_fresh_repo|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|en|
|Size:|1.6 GB|

## References

https://huggingface.co/tom-010/judge_answer__22_deberta_v3_large_fresh_repo