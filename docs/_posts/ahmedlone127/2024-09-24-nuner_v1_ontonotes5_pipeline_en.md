---
layout: model
title: English nuner_v1_ontonotes5_pipeline pipeline RoBertaForTokenClassification from guishe
author: John Snow Labs
name: nuner_v1_ontonotes5_pipeline
date: 2024-09-24
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`nuner_v1_ontonotes5_pipeline` is a English model originally trained by guishe.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nuner_v1_ontonotes5_pipeline_en_5.5.0_3.0_1727139860961.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/nuner_v1_ontonotes5_pipeline_en_5.5.0_3.0_1727139860961.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("nuner_v1_ontonotes5_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("nuner_v1_ontonotes5_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nuner_v1_ontonotes5_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|453.8 MB|

## References

https://huggingface.co/guishe/nuner-v1_ontonotes5

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification