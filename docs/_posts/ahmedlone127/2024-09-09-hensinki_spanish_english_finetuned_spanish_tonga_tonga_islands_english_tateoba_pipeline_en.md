---
layout: model
title: English hensinki_spanish_english_finetuned_spanish_tonga_tonga_islands_english_tateoba_pipeline pipeline MarianTransformer from beanslmao
author: John Snow Labs
name: hensinki_spanish_english_finetuned_spanish_tonga_tonga_islands_english_tateoba_pipeline
date: 2024-09-09
tags: [en, open_source, pipeline, onnx]
task: Translation
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`hensinki_spanish_english_finetuned_spanish_tonga_tonga_islands_english_tateoba_pipeline` is a English model originally trained by beanslmao.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/hensinki_spanish_english_finetuned_spanish_tonga_tonga_islands_english_tateoba_pipeline_en_5.5.0_3.0_1725840056677.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/hensinki_spanish_english_finetuned_spanish_tonga_tonga_islands_english_tateoba_pipeline_en_5.5.0_3.0_1725840056677.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("hensinki_spanish_english_finetuned_spanish_tonga_tonga_islands_english_tateoba_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("hensinki_spanish_english_finetuned_spanish_tonga_tonga_islands_english_tateoba_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|hensinki_spanish_english_finetuned_spanish_tonga_tonga_islands_english_tateoba_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|539.8 MB|

## References

https://huggingface.co/beanslmao/hensinki-es-en-finetuned-spanish-to-english-tateoba

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer