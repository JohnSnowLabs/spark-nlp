---
layout: model
title: English bge_base_english_v1_5_course_recommender_v4python_pipeline pipeline BGEEmbeddings from datasocietyco
author: John Snow Labs
name: bge_base_english_v1_5_course_recommender_v4python_pipeline
date: 2024-12-18
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained BGEEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bge_base_english_v1_5_course_recommender_v4python_pipeline` is a English model originally trained by datasocietyco.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bge_base_english_v1_5_course_recommender_v4python_pipeline_en_5.5.1_3.0_1734563260689.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bge_base_english_v1_5_course_recommender_v4python_pipeline_en_5.5.1_3.0_1734563260689.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bge_base_english_v1_5_course_recommender_v4python_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bge_base_english_v1_5_course_recommender_v4python_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bge_base_english_v1_5_course_recommender_v4python_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|375.9 MB|

## References

https://huggingface.co/datasocietyco/bge-base-en-v1.5-course-recommender-v4python

## Included Models

- DocumentAssembler
- BGEEmbeddings