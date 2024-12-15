---
layout: model
title: English scenario_tcr_4_data_english_cardiff_eng_only4_pipeline pipeline XlmRoBertaForSequenceClassification from haryoaw
author: John Snow Labs
name: scenario_tcr_4_data_english_cardiff_eng_only4_pipeline
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`scenario_tcr_4_data_english_cardiff_eng_only4_pipeline` is a English model originally trained by haryoaw.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/scenario_tcr_4_data_english_cardiff_eng_only4_pipeline_en_5.5.1_3.0_1734292625209.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/scenario_tcr_4_data_english_cardiff_eng_only4_pipeline_en_5.5.1_3.0_1734292625209.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("scenario_tcr_4_data_english_cardiff_eng_only4_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("scenario_tcr_4_data_english_cardiff_eng_only4_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|scenario_tcr_4_data_english_cardiff_eng_only4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|821.2 MB|

## References

https://huggingface.co/haryoaw/scenario-TCR-4_data-en-cardiff_eng_only4

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification