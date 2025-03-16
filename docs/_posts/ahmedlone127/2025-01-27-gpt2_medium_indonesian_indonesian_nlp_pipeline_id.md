---
layout: model
title: Indonesian gpt2_medium_indonesian_indonesian_nlp_pipeline pipeline GPT2Transformer from indonesian-nlp
author: John Snow Labs
name: gpt2_medium_indonesian_indonesian_nlp_pipeline
date: 2025-01-27
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

Pretrained GPT2Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`gpt2_medium_indonesian_indonesian_nlp_pipeline` is a Indonesian model originally trained by indonesian-nlp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/gpt2_medium_indonesian_indonesian_nlp_pipeline_id_5.5.1_3.0_1737997776491.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/gpt2_medium_indonesian_indonesian_nlp_pipeline_id_5.5.1_3.0_1737997776491.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("gpt2_medium_indonesian_indonesian_nlp_pipeline", lang = "id")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("gpt2_medium_indonesian_indonesian_nlp_pipeline", lang = "id")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|gpt2_medium_indonesian_indonesian_nlp_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|id|
|Size:|1.3 GB|

## References

https://huggingface.co/indonesian-nlp/gpt2-medium-indonesian

## Included Models

- DocumentAssembler
- GPT2Transformer