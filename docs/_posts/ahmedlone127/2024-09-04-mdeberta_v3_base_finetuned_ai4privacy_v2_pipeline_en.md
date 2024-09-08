---
layout: model
title: English mdeberta_v3_base_finetuned_ai4privacy_v2_pipeline pipeline DeBertaForTokenClassification from Isotonic
author: John Snow Labs
name: mdeberta_v3_base_finetuned_ai4privacy_v2_pipeline
date: 2024-09-04
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
language: en
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained DeBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`mdeberta_v3_base_finetuned_ai4privacy_v2_pipeline` is a English model originally trained by Isotonic.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/mdeberta_v3_base_finetuned_ai4privacy_v2_pipeline_en_5.5.0_3.0_1725472931903.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/mdeberta_v3_base_finetuned_ai4privacy_v2_pipeline_en_5.5.0_3.0_1725472931903.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("mdeberta_v3_base_finetuned_ai4privacy_v2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("mdeberta_v3_base_finetuned_ai4privacy_v2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|mdeberta_v3_base_finetuned_ai4privacy_v2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|890.8 MB|

## References

https://huggingface.co/Isotonic/mdeberta-v3-base_finetuned_ai4privacy_v2

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForTokenClassification