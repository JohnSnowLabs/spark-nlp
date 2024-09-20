---
layout: model
title: English epc_adept_roberta_l_all_pipeline pipeline RoBertaForSequenceClassification from veronica320
author: John Snow Labs
name: epc_adept_roberta_l_all_pipeline
date: 2024-09-16
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`epc_adept_roberta_l_all_pipeline` is a English model originally trained by veronica320.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/epc_adept_roberta_l_all_pipeline_en_5.5.0_3.0_1726527095417.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/epc_adept_roberta_l_all_pipeline_en_5.5.0_3.0_1726527095417.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("epc_adept_roberta_l_all_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("epc_adept_roberta_l_all_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|epc_adept_roberta_l_all_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|1.3 GB|

## References

https://huggingface.co/veronica320/EPC_ADEPT_roberta-l_all

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification