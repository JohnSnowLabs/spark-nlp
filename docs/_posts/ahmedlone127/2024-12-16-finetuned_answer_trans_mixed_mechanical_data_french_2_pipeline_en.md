---
layout: model
title: English finetuned_answer_trans_mixed_mechanical_data_french_2_pipeline pipeline T5Transformer from amiraMamdouh
author: John Snow Labs
name: finetuned_answer_trans_mixed_mechanical_data_french_2_pipeline
date: 2024-12-16
tags: [en, open_source, pipeline, onnx]
task: [Question Answering, Summarization, Translation, Text Generation]
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

Pretrained T5Transformer, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`finetuned_answer_trans_mixed_mechanical_data_french_2_pipeline` is a English model originally trained by amiraMamdouh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/finetuned_answer_trans_mixed_mechanical_data_french_2_pipeline_en_5.5.1_3.0_1734327502780.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/finetuned_answer_trans_mixed_mechanical_data_french_2_pipeline_en_5.5.1_3.0_1734327502780.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("finetuned_answer_trans_mixed_mechanical_data_french_2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("finetuned_answer_trans_mixed_mechanical_data_french_2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|finetuned_answer_trans_mixed_mechanical_data_french_2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|347.7 MB|

## References

https://huggingface.co/amiraMamdouh/finetuned_answer_trans_mixed_mechanical_data_fr_2

## Included Models

- DocumentAssembler
- T5Transformer