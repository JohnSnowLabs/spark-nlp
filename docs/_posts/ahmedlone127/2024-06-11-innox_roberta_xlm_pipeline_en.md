---
layout: model
title: English innox_roberta_xlm_pipeline pipeline XlmRoBertaForTokenClassification from brao
author: John Snow Labs
name: innox_roberta_xlm_pipeline
date: 2024-06-11
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.4.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`innox_roberta_xlm_pipeline` is a English model originally trained by brao.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/innox_roberta_xlm_pipeline_en_5.4.0_3.0_1718113971358.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/innox_roberta_xlm_pipeline_en_5.4.0_3.0_1718113971358.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("innox_roberta_xlm_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("innox_roberta_xlm_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|innox_roberta_xlm_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|844.9 MB|

## References

https://huggingface.co/brao/innox-roberta-xlm

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification