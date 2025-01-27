---
layout: model
title: Ukrainian title_instruction_pipeline pipeline XlmRoBertaForSequenceClassification from zeusfsx
author: John Snow Labs
name: title_instruction_pipeline
date: 2025-01-26
tags: [uk, open_source, pipeline, onnx]
task: Text Classification
language: uk
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`title_instruction_pipeline` is a Ukrainian model originally trained by zeusfsx.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/title_instruction_pipeline_uk_5.5.1_3.0_1737881431559.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/title_instruction_pipeline_uk_5.5.1_3.0_1737881431559.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("title_instruction_pipeline", lang = "uk")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("title_instruction_pipeline", lang = "uk")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|title_instruction_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|uk|
|Size:|390.4 MB|

## References

https://huggingface.co/zeusfsx/title-instruction

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification