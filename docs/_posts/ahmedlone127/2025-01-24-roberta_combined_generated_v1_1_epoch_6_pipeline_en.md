---
layout: model
title: English roberta_combined_generated_v1_1_epoch_6_pipeline pipeline RoBertaForTokenClassification from ICT2214Team7
author: John Snow Labs
name: roberta_combined_generated_v1_1_epoch_6_pipeline
date: 2025-01-24
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_combined_generated_v1_1_epoch_6_pipeline` is a English model originally trained by ICT2214Team7.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_combined_generated_v1_1_epoch_6_pipeline_en_5.5.1_3.0_1737702592470.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_combined_generated_v1_1_epoch_6_pipeline_en_5.5.1_3.0_1737702592470.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_combined_generated_v1_1_epoch_6_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_combined_generated_v1_1_epoch_6_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_combined_generated_v1_1_epoch_6_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|306.6 MB|

## References

https://huggingface.co/ICT2214Team7/RoBERTa_Combined_Generated_v1.1_epoch_6

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification