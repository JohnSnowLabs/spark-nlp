---
layout: model
title: Bengali bengali_lyricist_gpt2 GPT2Transformer from khalidsaifullaah
author: John Snow Labs
name: bengali_lyricist_gpt2
date: 2025-04-04
tags: [bn, open_source, onnx, text_generation, gpt2]
task: [Question Answering, Summarization, Translation, Text Generation]
language: bn
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

Pretrained GPT2Transformer model, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bengali_lyricist_gpt2` is a Bengali model originally trained by khalidsaifullaah.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bengali_lyricist_gpt2_bn_5.5.1_3.0_1743795437248.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bengali_lyricist_gpt2_bn_5.5.1_3.0_1743795437248.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")

seq2seq = GPT2Transformer.pretrained("bengali_lyricist_gpt2","bn") \
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
    
val seq2seq = GPT2Transformer.pretrained("bengali_lyricist_gpt2","bn") 
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
|Model Name:|bengali_lyricist_gpt2|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document]|
|Output Labels:|[generation]|
|Language:|bn|
|Size:|467.2 MB|

## References

https://huggingface.co/khalidsaifullaah/bengali-lyricist-gpt2