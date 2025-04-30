---
layout: model
title: English mdeberta_v3_base_finetuned_french_green_stance_supportive_pipeline pipeline DeBertaForSequenceClassification from mljn
author: John Snow Labs
name: mdeberta_v3_base_finetuned_french_green_stance_supportive_pipeline
date: 2025-03-28
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mdeberta_v3_base_finetuned_french_green_stance_supportive_pipeline` is a English model originally trained by mljn.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mdeberta_v3_base_finetuned_french_green_stance_supportive_pipeline_en_5.5.1_3.0_1743130343800.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mdeberta_v3_base_finetuned_french_green_stance_supportive_pipeline_en_5.5.1_3.0_1743130343800.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mdeberta_v3_base_finetuned_french_green_stance_supportive_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mdeberta_v3_base_finetuned_french_green_stance_supportive_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mdeberta_v3_base_finetuned_french_green_stance_supportive_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|855.4 MB|

## References

https://huggingface.co/mljn/mdeberta-v3-base-finetuned-french-green-stance-supportive

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification