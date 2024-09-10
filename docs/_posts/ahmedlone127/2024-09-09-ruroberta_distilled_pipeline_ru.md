---
layout: model
title: Russian ruroberta_distilled_pipeline pipeline RoBertaEmbeddings from d0rj
author: John Snow Labs
name: ruroberta_distilled_pipeline
date: 2024-09-09
tags: [ru, open_source, pipeline, onnx]
task: Embeddings
language: ru
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`ruroberta_distilled_pipeline` is a Russian model originally trained by d0rj.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/ruroberta_distilled_pipeline_ru_5.5.0_3.0_1725910188761.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/ruroberta_distilled_pipeline_ru_5.5.0_3.0_1725910188761.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("ruroberta_distilled_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("ruroberta_distilled_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ruroberta_distilled_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|432.0 MB|

## References

https://huggingface.co/d0rj/ruRoberta-distilled

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings