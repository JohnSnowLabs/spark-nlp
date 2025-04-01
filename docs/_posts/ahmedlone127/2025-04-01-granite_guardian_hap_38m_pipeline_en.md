---
layout: model
title: English granite_guardian_hap_38m_pipeline pipeline RoBertaForSequenceClassification from ibm-granite
author: John Snow Labs
name: granite_guardian_hap_38m_pipeline
date: 2025-04-01
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`granite_guardian_hap_38m_pipeline` is a English model originally trained by ibm-granite.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/granite_guardian_hap_38m_pipeline_en_5.5.1_3.0_1743486660720.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/granite_guardian_hap_38m_pipeline_en_5.5.1_3.0_1743486660720.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("granite_guardian_hap_38m_pipeline", lang = "en")
annotations =  pipeline.transform(df)
```
```scala
val pipeline = new PretrainedPipeline("granite_guardian_hap_38m_pipeline", lang = "en")
val annotations = pipeline.transform(df)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|granite_guardian_hap_38m_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|145.4 MB|

## References

References

https://huggingface.co/ibm-granite/granite-guardian-hap-38m

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification