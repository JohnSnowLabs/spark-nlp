---
layout: model
title: English cnn_dailymail_t5_small_fine_tuning_500_10_3000_6_e_1_s6789_v4_l6_pipeline pipeline T5Transformer from KingKazma
author: John Snow Labs
name: cnn_dailymail_t5_small_fine_tuning_500_10_3000_6_e_1_s6789_v4_l6_pipeline
date: 2024-08-20
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cnn_dailymail_t5_small_fine_tuning_500_10_3000_6_e_1_s6789_v4_l6_pipeline` is a English model originally trained by KingKazma.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cnn_dailymail_t5_small_fine_tuning_500_10_3000_6_e_1_s6789_v4_l6_pipeline_en_5.4.2_3.0_1724118465835.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cnn_dailymail_t5_small_fine_tuning_500_10_3000_6_e_1_s6789_v4_l6_pipeline_en_5.4.2_3.0_1724118465835.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cnn_dailymail_t5_small_fine_tuning_500_10_3000_6_e_1_s6789_v4_l6_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cnn_dailymail_t5_small_fine_tuning_500_10_3000_6_e_1_s6789_v4_l6_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cnn_dailymail_t5_small_fine_tuning_500_10_3000_6_e_1_s6789_v4_l6_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|301.9 MB|

## References

https://huggingface.co/KingKazma/cnn_dailymail_t5-small_fine_tuning_500_10_3000_6_e-1_s6789_v4_l6

## Included Models

- DocumentAssembler
- T5Transformer