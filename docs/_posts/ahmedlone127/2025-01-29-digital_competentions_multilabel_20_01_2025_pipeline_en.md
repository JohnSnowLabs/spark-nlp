---
layout: model
title: English digital_competentions_multilabel_20_01_2025_pipeline pipeline XlmRoBertaForSequenceClassification from EvgenyBondarenko
author: John Snow Labs
name: digital_competentions_multilabel_20_01_2025_pipeline
date: 2025-01-29
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`digital_competentions_multilabel_20_01_2025_pipeline` is a English model originally trained by EvgenyBondarenko.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/digital_competentions_multilabel_20_01_2025_pipeline_en_5.5.1_3.0_1738178291298.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/digital_competentions_multilabel_20_01_2025_pipeline_en_5.5.1_3.0_1738178291298.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("digital_competentions_multilabel_20_01_2025_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("digital_competentions_multilabel_20_01_2025_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|digital_competentions_multilabel_20_01_2025_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/EvgenyBondarenko/digital_competentions_multilabel_20_01_2025

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification