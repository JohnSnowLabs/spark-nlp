---
layout: model
title: Mongolian mongolian_xlm_roberta_base_named_entity_pipeline pipeline XlmRoBertaForTokenClassification from 2rtl3
author: John Snow Labs
name: mongolian_xlm_roberta_base_named_entity_pipeline
date: 2024-09-02
tags: [mn, open_source, pipeline, onnx]
task: Named Entity Recognition
language: mn
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mongolian_xlm_roberta_base_named_entity_pipeline` is a Mongolian model originally trained by 2rtl3.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mongolian_xlm_roberta_base_named_entity_pipeline_mn_5.5.0_3.0_1725308817705.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mongolian_xlm_roberta_base_named_entity_pipeline_mn_5.5.0_3.0_1725308817705.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mongolian_xlm_roberta_base_named_entity_pipeline", lang = "mn")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mongolian_xlm_roberta_base_named_entity_pipeline", lang = "mn")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mongolian_xlm_roberta_base_named_entity_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|mn|
|Size:|842.2 MB|

## References

https://huggingface.co/2rtl3/mn-xlm-roberta-base-named-entity

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification