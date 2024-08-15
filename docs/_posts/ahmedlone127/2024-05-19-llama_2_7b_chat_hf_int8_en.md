---
layout: model
title: Llama-2 text-to-text model 7b int8
author: John Snow Labs
name: llama_2_7b_chat_hf_int8
date: 2024-05-19
tags: [en, llama2, open_source]
task: Text Generation
language: en
nav_key: models
edition: Spark NLP 5.3.0
spark_version: 3.0
supported: true
recommended: true
annotator: LLAMA2Transformer
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Meta developed and publicly released the Llama 2 family of large language models (LLMs), a collection of pretrained and fine-tuned generative text models ranging in scale from 7 billion to 70 billion parameters. Our fine-tuned LLMs, called Llama-2-Chat, are optimized for dialogue use cases. Llama-2-Chat models outperform open-source chat models on most benchmarks we tested, and in our human evaluations for helpfulness and safety, are on par with some popular closed-source models like ChatGPT and PaLM.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/llama_2_7b_chat_hf_int8_en_5.3.0_3.0_1708952065310.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/llama_2_7b_chat_hf_int8_en_5.3.0_3.0_1708952065310.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler() \
.setInputCol("text") \
.setOutputCol("documents")

llama2 = LLAMA2Transformer \
    .pretrained("llama_2_7b_chat_hf_int8") \
    .setMaxOutputLength(50) \
    .setDoSample(False) \
    .setInputCols(["documents"]) \
    .setOutputCol("generation")

pipeline = Pipeline().setStages([documentAssembler, llama2])
data = spark.createDataFrame([["My name is Leonardo."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.select("summaries.generation").show(truncate=False)
```
```scala
val documentAssembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("documents")

val llama2 = LLAMA2Transformer.pretrained("llama_2_7b_chat_hf_int8") 
    .setMaxOutputLength(50) 
    .setDoSample(False) 
    .setInputCols(["documents"]) 
    .setOutputCol("generation")

val pipeline = new Pipeline().setStages(Array(documentAssembler, llama2))

val data = Seq("My name is Leonardo.").toDF("text")
val result = pipeline.fit(data).transform(data)
results.select("generation.result").show(truncate = false)
```

</div>


{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|llama_2_7b_chat_hf_int8|
|Compatibility:|Spark NLP 5.3.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[documents]|
|Output Labels:|[generation]|
|Language:|en|
