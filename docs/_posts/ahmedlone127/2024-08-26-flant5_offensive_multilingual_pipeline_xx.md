---
layout: model
title: Multilingual flant5_offensive_multilingual_pipeline pipeline T5Transformer from JenniferHJF
author: John Snow Labs
name: flant5_offensive_multilingual_pipeline
date: 2024-08-26
tags: [xx, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: xx
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`flant5_offensive_multilingual_pipeline` is a Multilingual model originally trained by JenniferHJF.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/flant5_offensive_multilingual_pipeline_xx_5.4.2_3.0_1724644436356.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/flant5_offensive_multilingual_pipeline_xx_5.4.2_3.0_1724644436356.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("flant5_offensive_multilingual_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("flant5_offensive_multilingual_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|flant5_offensive_multilingual_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|1.0 GB|

## References

https://huggingface.co/JenniferHJF/Flant5-offensive-multilingual

## Included Models

- DocumentAssembler
- T5Transformer