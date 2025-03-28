---
layout: model
title: Uzbek xlm_roberta_base_lowercase_high_accuracy_pipeline pipeline XlmRoBertaForTokenClassification from tukhtashevshohruh
author: John Snow Labs
name: xlm_roberta_base_lowercase_high_accuracy_pipeline
date: 2025-03-28
tags: [uz, open_source, pipeline, onnx]
task: Named Entity Recognition
language: uz
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlm_roberta_base_lowercase_high_accuracy_pipeline` is a Uzbek model originally trained by tukhtashevshohruh.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_lowercase_high_accuracy_pipeline_uz_5.5.1_3.0_1743120412924.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlm_roberta_base_lowercase_high_accuracy_pipeline_uz_5.5.1_3.0_1743120412924.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlm_roberta_base_lowercase_high_accuracy_pipeline", lang = "uz")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlm_roberta_base_lowercase_high_accuracy_pipeline", lang = "uz")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_base_lowercase_high_accuracy_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|uz|
|Size:|839.4 MB|

## References

https://huggingface.co/tukhtashevshohruh/xlm-roberta-base-lowercase-high-accuracy

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForTokenClassification