---
layout: model
title: English burmese_translation_helsinki2_pipeline pipeline MarianTransformer from duwuonline
author: John Snow Labs
name: burmese_translation_helsinki2_pipeline
date: 2024-09-17
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

Pretrained MarianTransformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`burmese_translation_helsinki2_pipeline` is a English model originally trained by duwuonline.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/burmese_translation_helsinki2_pipeline_en_5.5.0_3.0_1726532850843.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/burmese_translation_helsinki2_pipeline_en_5.5.0_3.0_1726532850843.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("burmese_translation_helsinki2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("burmese_translation_helsinki2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|burmese_translation_helsinki2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|475.3 MB|

## References

https://huggingface.co/duwuonline/my-translation-helsinki2

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- MarianTransformer