---
layout: model
title: Russian medrurobertalarge_pipeline pipeline RoBertaEmbeddings from DmitryPogrebnoy
author: John Snow Labs
name: medrurobertalarge_pipeline
date: 2024-09-03
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`medrurobertalarge_pipeline` is a Russian model originally trained by DmitryPogrebnoy.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/medrurobertalarge_pipeline_ru_5.5.0_3.0_1725375743433.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/medrurobertalarge_pipeline_ru_5.5.0_3.0_1725375743433.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("medrurobertalarge_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("medrurobertalarge_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|medrurobertalarge_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|1.3 GB|

## References

https://huggingface.co/DmitryPogrebnoy/MedRuRobertaLarge

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings