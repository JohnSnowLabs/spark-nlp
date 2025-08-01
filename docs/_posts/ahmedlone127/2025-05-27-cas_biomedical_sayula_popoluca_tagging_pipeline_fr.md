---
layout: model
title: French cas_biomedical_sayula_popoluca_tagging_pipeline pipeline CamemBertForTokenClassification from Dr-BERT
author: John Snow Labs
name: cas_biomedical_sayula_popoluca_tagging_pipeline
date: 2025-05-27
tags: [fr, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fr
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cas_biomedical_sayula_popoluca_tagging_pipeline` is a French model originally trained by Dr-BERT.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cas_biomedical_sayula_popoluca_tagging_pipeline_fr_5.5.1_3.0_1748370374667.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cas_biomedical_sayula_popoluca_tagging_pipeline_fr_5.5.1_3.0_1748370374667.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cas_biomedical_sayula_popoluca_tagging_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cas_biomedical_sayula_popoluca_tagging_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cas_biomedical_sayula_popoluca_tagging_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|412.7 MB|

## References

https://huggingface.co/Dr-BERT/CAS-Biomedical-POS-Tagging

## Included Models

- DocumentAssembler
- TokenizerModel
- CamemBertForTokenClassification