---
layout: model
title: French autonlp_fr_another_test_565016091 CamemBertForSequenceClassification from medA
author: John Snow Labs
name: autonlp_fr_another_test_565016091
date: 2024-01-19
tags: [camembert, fr, open_source, sequence_classification, onnx]
task: Text Classification
language: fr
edition: Spark NLP 5.2.4
spark_version: 3.0
supported: true
engine: onnx
annotator: CamemBertForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`autonlp_fr_another_test_565016091` is a French model originally trained by medA.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/autonlp_fr_another_test_565016091_fr_5.2.4_3.0_1705701832503.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/autonlp_fr_another_test_565016091_fr_5.2.4_3.0_1705701832503.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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
    
sequenceClassifier = CamemBertForSequenceClassification.pretrained("autonlp_fr_another_test_565016091","fr")\
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
    
val sequenceClassifier = CamemBertForSequenceClassification.pretrained("autonlp_fr_another_test_565016091","fr")
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
|Model Name:|autonlp_fr_another_test_565016091|
|Compatibility:|Spark NLP 5.2.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents, token]|
|Output Labels:|[class]|
|Language:|fr|
|Size:|410.3 MB|

## References

https://huggingface.co/medA/autonlp-FR_another_test-565016091