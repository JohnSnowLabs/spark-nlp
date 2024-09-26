---
layout: model
title: Persian parsbert_base_parsinlu_entailment_pipeline pipeline BertForSequenceClassification from persiannlp
author: John Snow Labs
name: parsbert_base_parsinlu_entailment_pipeline
date: 2024-09-26
tags: [fa, open_source, pipeline, onnx]
task: Text Classification
language: fa
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`parsbert_base_parsinlu_entailment_pipeline` is a Persian model originally trained by persiannlp.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/parsbert_base_parsinlu_entailment_pipeline_fa_5.5.0_3.0_1727313917653.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/parsbert_base_parsinlu_entailment_pipeline_fa_5.5.0_3.0_1727313917653.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("parsbert_base_parsinlu_entailment_pipeline", lang = "fa")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("parsbert_base_parsinlu_entailment_pipeline", lang = "fa")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|parsbert_base_parsinlu_entailment_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|fa|
|Size:|608.8 MB|

## References

https://huggingface.co/persiannlp/parsbert-base-parsinlu-entailment

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification