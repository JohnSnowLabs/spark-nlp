---
layout: model
title: French esgi_nlp_tp4_virtual_assistant_pipeline_pipeline pipeline RoBertaForTokenClassification from Florent-COMPAGNONI
author: John Snow Labs
name: esgi_nlp_tp4_virtual_assistant_pipeline_pipeline
date: 2024-09-06
tags: [fr, open_source, pipeline, onnx]
task: Named Entity Recognition
language: fr
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`esgi_nlp_tp4_virtual_assistant_pipeline_pipeline` is a French model originally trained by Florent-COMPAGNONI.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/esgi_nlp_tp4_virtual_assistant_pipeline_pipeline_fr_5.5.0_3.0_1725638349055.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/esgi_nlp_tp4_virtual_assistant_pipeline_pipeline_fr_5.5.0_3.0_1725638349055.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("esgi_nlp_tp4_virtual_assistant_pipeline_pipeline", lang = "fr")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("esgi_nlp_tp4_virtual_assistant_pipeline_pipeline", lang = "fr")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|esgi_nlp_tp4_virtual_assistant_pipeline_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fr|
|Size:|317.5 MB|

## References

https://huggingface.co/Florent-COMPAGNONI/esgi-nlp-tp4-virtual_assistant_pipeline

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification