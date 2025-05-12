---
layout: model
title: English shqa_ft_p3_unifiedqa_musique_pipeline pipeline T5Transformer from abiantonio
author: John Snow Labs
name: shqa_ft_p3_unifiedqa_musique_pipeline
date: 2025-02-06
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`shqa_ft_p3_unifiedqa_musique_pipeline` is a English model originally trained by abiantonio.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/shqa_ft_p3_unifiedqa_musique_pipeline_en_5.5.1_3.0_1738807881732.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/shqa_ft_p3_unifiedqa_musique_pipeline_en_5.5.1_3.0_1738807881732.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("shqa_ft_p3_unifiedqa_musique_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("shqa_ft_p3_unifiedqa_musique_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|shqa_ft_p3_unifiedqa_musique_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|923.9 MB|

## References

https://huggingface.co/abiantonio/shqa-ft-p3-unifiedqa-musique

## Included Models

- DocumentAssembler
- T5Transformer