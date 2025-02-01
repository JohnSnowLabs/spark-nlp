---
layout: model
title: English roberta_large_lemon_spell_5k_pipeline pipeline RoBertaForTokenClassification from manred1997
author: John Snow Labs
name: roberta_large_lemon_spell_5k_pipeline
date: 2025-01-31
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_large_lemon_spell_5k_pipeline` is a English model originally trained by manred1997.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_large_lemon_spell_5k_pipeline_en_5.5.1_3.0_1738366588618.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_large_lemon_spell_5k_pipeline_en_5.5.1_3.0_1738366588618.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("roberta_large_lemon_spell_5k_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("roberta_large_lemon_spell_5k_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_large_lemon_spell_5k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

References

https://huggingface.co/manred1997/roberta-large_lemon-spell_5k

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification