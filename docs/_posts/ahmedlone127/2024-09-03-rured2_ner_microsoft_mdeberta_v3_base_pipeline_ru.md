---
layout: model
title: Russian rured2_ner_microsoft_mdeberta_v3_base_pipeline pipeline DeBertaForTokenClassification from denis-gordeev
author: John Snow Labs
name: rured2_ner_microsoft_mdeberta_v3_base_pipeline
date: 2024-09-03
tags: [ru, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained DeBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`rured2_ner_microsoft_mdeberta_v3_base_pipeline` is a Russian model originally trained by denis-gordeev.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/rured2_ner_microsoft_mdeberta_v3_base_pipeline_ru_5.5.0_3.0_1725400617764.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/rured2_ner_microsoft_mdeberta_v3_base_pipeline_ru_5.5.0_3.0_1725400617764.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("rured2_ner_microsoft_mdeberta_v3_base_pipeline", lang = "ru")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("rured2_ner_microsoft_mdeberta_v3_base_pipeline", lang = "ru")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|rured2_ner_microsoft_mdeberta_v3_base_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|ru|
|Size:|825.1 MB|

## References

https://huggingface.co/denis-gordeev/rured2-ner-microsoft-mdeberta-v3-base

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForTokenClassification