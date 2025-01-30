---
layout: model
title: English partyclim_hp_norwegian_processing_at_all_adjusted_tonga_tonga_islands_justify_pipeline pipeline XlmRoBertaForSequenceClassification from julitudo
author: John Snow Labs
name: partyclim_hp_norwegian_processing_at_all_adjusted_tonga_tonga_islands_justify_pipeline
date: 2025-01-26
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`partyclim_hp_norwegian_processing_at_all_adjusted_tonga_tonga_islands_justify_pipeline` is a English model originally trained by julitudo.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/partyclim_hp_norwegian_processing_at_all_adjusted_tonga_tonga_islands_justify_pipeline_en_5.5.1_3.0_1737881907473.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/partyclim_hp_norwegian_processing_at_all_adjusted_tonga_tonga_islands_justify_pipeline_en_5.5.1_3.0_1737881907473.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("partyclim_hp_norwegian_processing_at_all_adjusted_tonga_tonga_islands_justify_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("partyclim_hp_norwegian_processing_at_all_adjusted_tonga_tonga_islands_justify_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|partyclim_hp_norwegian_processing_at_all_adjusted_tonga_tonga_islands_justify_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|812.4 MB|

## References

https://huggingface.co/julitudo/partyclim_HP_no_processing_at_all_adjusted_to_justify

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification