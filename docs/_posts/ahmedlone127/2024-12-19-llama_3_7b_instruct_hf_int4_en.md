---
layout: model
title: llama_3_7b_instruct_hf_int4 model from meta-llama
author: John Snow Labs
name: llama_3_7b_instruct_hf_int4
date: 2024-12-19
tags: [en, open_source, openvino]
task: Text Generation
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
engine: openvino
annotator: LLAMA3Transformer
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained LLAMA3Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`llama_3_7b_instruct_hf_int4` is a english model originally trained by meta-llama.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/llama_3_7b_instruct_hf_int4_en_5.5.1_3.0_1734607923506.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/llama_3_7b_instruct_hf_int4_en_5.5.1_3.0_1734607923506.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
 
documentAssembler = DocumentAssembler() \
      .setInputCol("text") \
      .setOutputCol("document")
    
seq2seq = LLAMA3Transformer.pretrained("llama_3_7b_instruct_hf_int4","en") \
      .setInputCols(["document"]) \
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
    
val seq2seq = LLAMA3Transformer.pretrained("llama_3_7b_instruct_hf_int4","en") 
    .setInputCols(Array("document")) 
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
|Model Name:|llama_3_7b_instruct_hf_int4|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[generation]|
|Language:|en|
|Size:|4.8 GB|

## References

https://huggingface.co/meta-llama/Llama-2-7b-chat-hf