---
layout: model
title: English infoxlm_base_on_custom_kural_500_pipeline pipeline XlmRoBertaForSequenceClassification from bikram22pi7
author: John Snow Labs
name: infoxlm_base_on_custom_kural_500_pipeline
date: 2024-09-20
tags: [en, open_source, pipeline, onnx]
task: Text Classification
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`infoxlm_base_on_custom_kural_500_pipeline` is a English model originally trained by bikram22pi7.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/infoxlm_base_on_custom_kural_500_pipeline_en_5.5.0_3.0_1726846462142.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/infoxlm_base_on_custom_kural_500_pipeline_en_5.5.0_3.0_1726846462142.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("infoxlm_base_on_custom_kural_500_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("infoxlm_base_on_custom_kural_500_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|infoxlm_base_on_custom_kural_500_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|777.7 MB|

## References

https://huggingface.co/bikram22pi7/infoxlm-base-on-custom-kural-500

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification