---
layout: model
title: Indonesian gpt2_small_indonesia_fine_tuning_poem_pipeline pipeline GPT2Transformer from ayameRushia
author: John Snow Labs
name: gpt2_small_indonesia_fine_tuning_poem_pipeline
date: 2025-04-02
tags: [id, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: id
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2_small_indonesia_fine_tuning_poem_pipeline` is a Indonesian model originally trained by ayameRushia.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2_small_indonesia_fine_tuning_poem_pipeline_id_5.5.1_3.0_1743568342940.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2_small_indonesia_fine_tuning_poem_pipeline_id_5.5.1_3.0_1743568342940.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2_small_indonesia_fine_tuning_poem_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2_small_indonesia_fine_tuning_poem_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2_small_indonesia_fine_tuning_poem_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|467.4 MB|

## References

https://huggingface.co/ayameRushia/gpt2-small-indonesia-fine-tuning-poem

## Included Models

- DocumentAssembler
- GPT2Transformer