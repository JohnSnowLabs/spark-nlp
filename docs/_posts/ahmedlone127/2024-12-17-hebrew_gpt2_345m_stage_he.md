---
layout: model
title: Hebrew hebrew_gpt2_345m_stage GPT2Transformer from Norod78
author: John Snow Labs
name: hebrew_gpt2_345m_stage
date: 2024-12-17
tags: [he, open_source, onnx, text_generation, gpt2]
task: [Question Answering, Summarization, Translation, Text Generation]
language: he
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: onnx
annotator: GPT2Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hebrew_gpt2_345m_stage` is a Hebrew model originally trained by Norod78.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hebrew_gpt2_345m_stage_he_5.5.1_3.0_1734394026208.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hebrew_gpt2_345m_stage_he_5.5.1_3.0_1734394026208.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

seq2seq = GPT2Transformer.pretrained("hebrew_gpt2_345m_stage","he") \
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
    
val seq2seq = GPT2Transformer.pretrained("hebrew_gpt2_345m_stage","he") 
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
|Model Name:|hebrew_gpt2_345m_stage|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[generation]|
|Language:|he|
|Size:|1.5 GB|

## References

https://huggingface.co/Norod78/Hebrew-GPT2-345M-Stage