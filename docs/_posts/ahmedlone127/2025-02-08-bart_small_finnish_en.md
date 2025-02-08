---
layout: model
title: English bart_small_finnish BartTransformer from Finnish-NLP
author: John Snow Labs
name: bart_small_finnish
date: 2025-02-08
tags: [en, open_source, onnx, text_generation, bart]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: BartTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bart_small_finnish` is a English model originally trained by Finnish-NLP.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bart_small_finnish_en_5.5.1_3.0_1738973884653.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bart_small_finnish_en_5.5.1_3.0_1738973884653.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

seq2seq = BartTransformer.pretrained("bart_small_finnish","en") \
      .setInputCols(["documents"]) \
      .setOutputCol("generation")       
        
pipeline = Pipeline().setStages([documentAssembler, seq2seq])
data = spark.createDataFrame([["I love spark-nlp"]]).toDF("text")
pipelineModel = pipeline.fit(data)
pipelineDF = pipelineModel.transform(data)

```
```scala

val documentAssembler = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")
    
val seq2seq = BartTransformer.pretrained("bart_small_finnish","en") 
    .setInputCols(Array("documents")) 
    .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, seq2seq))
val data = Seq("I love spark-nlp").toDF("text")
val pipelineModel = pipeline.fit(data)
val pipelineDF = pipelineModel.transform(data)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bart_small_finnish|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|en|
|Size:|458.6 MB|

## References

https://huggingface.co/Finnish-NLP/bart-small-finnish