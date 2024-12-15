---
layout: model
title: English furina_seed42_eng_kinyarwanda_amh_cross_latin_2e_05_pipeline pipeline XlmRoBertaForSequenceClassification from Shijia
author: John Snow Labs
name: furina_seed42_eng_kinyarwanda_amh_cross_latin_2e_05_pipeline
date: 2024-12-15
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`furina_seed42_eng_kinyarwanda_amh_cross_latin_2e_05_pipeline` is a English model originally trained by Shijia.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/furina_seed42_eng_kinyarwanda_amh_cross_latin_2e_05_pipeline_en_5.5.1_3.0_1734242039847.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/furina_seed42_eng_kinyarwanda_amh_cross_latin_2e_05_pipeline_en_5.5.1_3.0_1734242039847.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("furina_seed42_eng_kinyarwanda_amh_cross_latin_2e_05_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("furina_seed42_eng_kinyarwanda_amh_cross_latin_2e_05_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|furina_seed42_eng_kinyarwanda_amh_cross_latin_2e_05_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.5 GB|

## References

https://huggingface.co/Shijia/furina_seed42_eng_kin_amh_cross_latin_2e-05

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification