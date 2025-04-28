---
layout: model
title: English groups_sayula_popoluca_neg_classification_28_2_pipeline pipeline RoBertaForTokenClassification from GiladH
author: John Snow Labs
name: groups_sayula_popoluca_neg_classification_28_2_pipeline
date: 2025-02-03
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`groups_sayula_popoluca_neg_classification_28_2_pipeline` is a English model originally trained by GiladH.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/groups_sayula_popoluca_neg_classification_28_2_pipeline_en_5.5.1_3.0_1738586406980.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/groups_sayula_popoluca_neg_classification_28_2_pipeline_en_5.5.1_3.0_1738586406980.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("groups_sayula_popoluca_neg_classification_28_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("groups_sayula_popoluca_neg_classification_28_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|groups_sayula_popoluca_neg_classification_28_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/GiladH/groups_pos_neg_classification_28_2

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification