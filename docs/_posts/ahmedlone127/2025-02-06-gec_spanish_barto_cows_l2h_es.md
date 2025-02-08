---
layout: model
title: Castilian, Spanish gec_spanish_barto_cows_l2h BartTransformer from SkitCon
author: John Snow Labs
name: gec_spanish_barto_cows_l2h
date: 2025-02-06
tags: [es, open_source, onnx, text_generation, bart]
task: [Question Answering, Summarization, Translation, Text Generation]
language: es
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

Pretrained BartTransformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gec_spanish_barto_cows_l2h` is a Castilian, Spanish model originally trained by SkitCon.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gec_spanish_barto_cows_l2h_es_5.5.1_3.0_1738837134360.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gec_spanish_barto_cows_l2h_es_5.5.1_3.0_1738837134360.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

seq2seq = BartTransformer.pretrained("gec_spanish_barto_cows_l2h","es") \
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
    
val seq2seq = BartTransformer.pretrained("gec_spanish_barto_cows_l2h","es") 
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
|Model Name:|gec_spanish_barto_cows_l2h|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[output]|
|Language:|es|
|Size:|810.3 MB|

## References

https://huggingface.co/SkitCon/gec-spanish-BARTO-COWS-L2H