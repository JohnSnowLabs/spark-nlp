---
layout: model
title: English calc_baseline_t5_large_anonym_repos_pipeline pipeline T5Transformer from anonym-repos
author: John Snow Labs
name: calc_baseline_t5_large_anonym_repos_pipeline
date: 2024-08-12
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`calc_baseline_t5_large_anonym_repos_pipeline` is a English model originally trained by anonym-repos.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/calc_baseline_t5_large_anonym_repos_pipeline_en_5.4.2_3.0_1723447521428.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/calc_baseline_t5_large_anonym_repos_pipeline_en_5.4.2_3.0_1723447521428.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("calc_baseline_t5_large_anonym_repos_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("calc_baseline_t5_large_anonym_repos_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|calc_baseline_t5_large_anonym_repos_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|3.0 GB|

## References

https://huggingface.co/anonym-repos/calc-baseline-t5-large

## Included Models

- DocumentAssembler
- T5Transformer