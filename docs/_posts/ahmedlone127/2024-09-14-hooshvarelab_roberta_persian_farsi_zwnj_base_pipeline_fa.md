---
layout: model
title: Persian hooshvarelab_roberta_persian_farsi_zwnj_base_pipeline pipeline RoBertaForTokenClassification from PerSpaCor
author: John Snow Labs
name: hooshvarelab_roberta_persian_farsi_zwnj_base_pipeline
date: 2024-09-14
tags: [fa, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fa
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hooshvarelab_roberta_persian_farsi_zwnj_base_pipeline` is a Persian model originally trained by PerSpaCor.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hooshvarelab_roberta_persian_farsi_zwnj_base_pipeline_fa_5.5.0_3.0_1726314945838.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hooshvarelab_roberta_persian_farsi_zwnj_base_pipeline_fa_5.5.0_3.0_1726314945838.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hooshvarelab_roberta_persian_farsi_zwnj_base_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hooshvarelab_roberta_persian_farsi_zwnj_base_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hooshvarelab_roberta_persian_farsi_zwnj_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|442.1 MB|

## References

https://huggingface.co/PerSpaCor/HooshvareLab-roberta-fa-zwnj-base

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification