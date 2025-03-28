---
layout: model
title: English dr_bertfinetuned_huge_19_pipeline pipeline CamemBertForTokenClassification from Amhyr
author: John Snow Labs
name: dr_bertfinetuned_huge_19_pipeline
date: 2025-03-28
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained CamemBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`dr_bertfinetuned_huge_19_pipeline` is a English model originally trained by Amhyr.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/dr_bertfinetuned_huge_19_pipeline_en_5.5.1_3.0_1743162734404.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/dr_bertfinetuned_huge_19_pipeline_en_5.5.1_3.0_1743162734404.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("dr_bertfinetuned_huge_19_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("dr_bertfinetuned_huge_19_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|dr_bertfinetuned_huge_19_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/Amhyr/dr-bertfinetuned-huge-19

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForTokenClassification