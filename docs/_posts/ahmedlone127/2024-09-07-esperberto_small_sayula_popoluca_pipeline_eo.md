---
layout: model
title: Esperanto esperberto_small_sayula_popoluca_pipeline pipeline RoBertaForTokenClassification from Xenova
author: John Snow Labs
name: esperberto_small_sayula_popoluca_pipeline
date: 2024-09-07
tags: [eo, open_source, pipeline, onnx]
task: Named Entity Recognition
language: eo
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`esperberto_small_sayula_popoluca_pipeline` is a Esperanto model originally trained by Xenova.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/esperberto_small_sayula_popoluca_pipeline_eo_5.5.0_3.0_1725668133672.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/esperberto_small_sayula_popoluca_pipeline_eo_5.5.0_3.0_1725668133672.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("esperberto_small_sayula_popoluca_pipeline", lang = "eo")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("esperberto_small_sayula_popoluca_pipeline", lang = "eo")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|esperberto_small_sayula_popoluca_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|eo|
|Size:|311.4 MB|

## References

https://huggingface.co/Xenova/EsperBERTo-small-pos

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification