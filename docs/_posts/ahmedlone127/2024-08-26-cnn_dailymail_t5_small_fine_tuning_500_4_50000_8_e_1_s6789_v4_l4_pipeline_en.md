---
layout: model
title: English cnn_dailymail_t5_small_fine_tuning_500_4_50000_8_e_1_s6789_v4_l4_pipeline pipeline T5Transformer from KingKazma
author: John Snow Labs
name: cnn_dailymail_t5_small_fine_tuning_500_4_50000_8_e_1_s6789_v4_l4_pipeline
date: 2024-08-26
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cnn_dailymail_t5_small_fine_tuning_500_4_50000_8_e_1_s6789_v4_l4_pipeline` is a English model originally trained by KingKazma.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cnn_dailymail_t5_small_fine_tuning_500_4_50000_8_e_1_s6789_v4_l4_pipeline_en_5.4.2_3.0_1724668334853.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cnn_dailymail_t5_small_fine_tuning_500_4_50000_8_e_1_s6789_v4_l4_pipeline_en_5.4.2_3.0_1724668334853.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cnn_dailymail_t5_small_fine_tuning_500_4_50000_8_e_1_s6789_v4_l4_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cnn_dailymail_t5_small_fine_tuning_500_4_50000_8_e_1_s6789_v4_l4_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cnn_dailymail_t5_small_fine_tuning_500_4_50000_8_e_1_s6789_v4_l4_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|348.6 MB|

## References

https://huggingface.co/KingKazma/cnn_dailymail_t5-small_fine_tuning_500_4_50000_8_e-1_s6789_v4_l4

## Included Models

- DocumentAssembler
- T5Transformer