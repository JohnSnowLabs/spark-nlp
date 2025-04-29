---
layout: model
title: English arxiv_summarization_fb_bart_base_2022_09_21_pipeline pipeline BartTransformer from farleyknight
author: John Snow Labs
name: arxiv_summarization_fb_bart_base_2022_09_21_pipeline
date: 2025-02-05
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BartTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`arxiv_summarization_fb_bart_base_2022_09_21_pipeline` is a English model originally trained by farleyknight.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/arxiv_summarization_fb_bart_base_2022_09_21_pipeline_en_5.5.1_3.0_1738723000480.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/arxiv_summarization_fb_bart_base_2022_09_21_pipeline_en_5.5.1_3.0_1738723000480.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("arxiv_summarization_fb_bart_base_2022_09_21_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("arxiv_summarization_fb_bart_base_2022_09_21_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|arxiv_summarization_fb_bart_base_2022_09_21_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|801.0 MB|

## References

https://huggingface.co/farleyknight/arxiv-summarization-fb-bart-base-2022-09-21

## Included Models

- DocumentAssembler
- BartTransformer