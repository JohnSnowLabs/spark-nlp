---
layout: model
title: English mobilebert_uncased_squad_v2_30_10_2_badokorach_pipeline pipeline BertForQuestionAnswering from badokorach
author: John Snow Labs
name: mobilebert_uncased_squad_v2_30_10_2_badokorach_pipeline
date: 2025-03-31
tags: [en, open_source, pipeline, onnx]
task: Question Answering
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

Pretrained BertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mobilebert_uncased_squad_v2_30_10_2_badokorach_pipeline` is a English model originally trained by badokorach.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mobilebert_uncased_squad_v2_30_10_2_badokorach_pipeline_en_5.5.1_3.0_1743458487759.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mobilebert_uncased_squad_v2_30_10_2_badokorach_pipeline_en_5.5.1_3.0_1743458487759.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mobilebert_uncased_squad_v2_30_10_2_badokorach_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mobilebert_uncased_squad_v2_30_10_2_badokorach_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mobilebert_uncased_squad_v2_30_10_2_badokorach_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|92.5 MB|

## References

https://huggingface.co/badokorach/mobilebert-uncased-squad-v2-30-10-2

## Included Models

- MultiDocumentAssembler
- BertForQuestionAnswering