---
layout: model
title: English flipped_5e_5_igbo_pipeline pipeline XlmRoBertaForTokenClassification from grace-pro
author: John Snow Labs
name: flipped_5e_5_igbo_pipeline
date: 2025-01-26
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

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`flipped_5e_5_igbo_pipeline` is a English model originally trained by grace-pro.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/flipped_5e_5_igbo_pipeline_en_5.5.1_3.0_1737853542397.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/flipped_5e_5_igbo_pipeline_en_5.5.1_3.0_1737853542397.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("flipped_5e_5_igbo_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("flipped_5e_5_igbo_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|flipped_5e_5_igbo_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.0 GB|

## References

https://huggingface.co/grace-pro/flipped_5e-5_igbo

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification