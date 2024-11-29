---
layout: model
title: nllb_distilled_600M_8int model from Facebook
author: John Snow Labs
name: nllb_distilled_600M_8int
date: 2024-11-27
tags: [en, open_source, pipeline, openvino, xx]
task: Text Generation
language: xx
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: NLLBTransformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained NLLBTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nllb_distilled_600M_8int` is a Multilingual model originally trained by facebook.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nllb_distilled_600M_8int_xx_5.5.1_3.0_1732741416718.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nllb_distilled_600M_8int_xx_5.5.1_3.0_1732741416718.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

seq2seq = NLLBTransformer.pretrained("mini_cpm_2b_8bit","xx") \
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
    
val seq2seq = NLLBTransformer.pretrained("mini_cpm_2b_8bit","xx") 
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
|Model Name:|nllb_distilled_600M_8int|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[generation]|
|Language:|xx|
|Size:|842.9 MB|

## References

https://huggingface.co/facebook/nllb-200-distilled-600M
