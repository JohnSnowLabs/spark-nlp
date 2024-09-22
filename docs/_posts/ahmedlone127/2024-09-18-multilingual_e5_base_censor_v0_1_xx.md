---
layout: model
title: Multilingual multilingual_e5_base_censor_v0_1 XlmRoBertaForSequenceClassification from Data-Lab
author: John Snow Labs
name: multilingual_e5_base_censor_v0_1
date: 2024-09-18
tags: [xx, open_source, onnx, sequence_classification, xlm_roberta]
task: Text Classification
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
engine: onnx
annotator: XlmRoBertaForSequenceClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`multilingual_e5_base_censor_v0_1` is a Multilingual model originally trained by Data-Lab.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/multilingual_e5_base_censor_v0_1_xx_5.5.0_3.0_1726659848298.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/multilingual_e5_base_censor_v0_1_xx_5.5.0_3.0_1726659848298.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

sequenceClassifier  = XlmRoBertaForSequenceClassification.pretrained("multilingual_e5_base_censor_v0_1","xx") \
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

val sequenceClassifier = XlmRoBertaForSequenceClassification.pretrained("multilingual_e5_base_censor_v0_1", "xx")
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
|Model Name:|multilingual_e5_base_censor_v0_1|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|xx|
|Size:|802.4 MB|

## References

https://huggingface.co/Data-Lab/multilingual-e5-base_censor_v0.1