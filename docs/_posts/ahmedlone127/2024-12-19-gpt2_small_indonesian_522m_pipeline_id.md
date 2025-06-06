---
layout: model
title: Indonesian gpt2_small_indonesian_522m_pipeline pipeline GPT2Transformer from cahya
author: John Snow Labs
name: gpt2_small_indonesian_522m_pipeline
date: 2024-12-19
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

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2_small_indonesian_522m_pipeline` is a Indonesian model originally trained by cahya.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2_small_indonesian_522m_pipeline_id_5.5.1_3.0_1734582609023.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2_small_indonesian_522m_pipeline_id_5.5.1_3.0_1734582609023.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2_small_indonesian_522m_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2_small_indonesian_522m_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2_small_indonesian_522m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|467.3 MB|

## References

https://huggingface.co/cahya/gpt2-small-indonesian-522M

## Included Models

- DocumentAssembler
- GPT2Transformer