---
layout: model
title: English question_v_statement_finetuned_roberta_basev2_pipeline pipeline RoBertaForSequenceClassification from mafwalter
author: John Snow Labs
name: question_v_statement_finetuned_roberta_basev2_pipeline
date: 2024-09-11
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

Pretrained RoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`question_v_statement_finetuned_roberta_basev2_pipeline` is a English model originally trained by mafwalter.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/question_v_statement_finetuned_roberta_basev2_pipeline_en_5.5.0_3.0_1726053853567.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/question_v_statement_finetuned_roberta_basev2_pipeline_en_5.5.0_3.0_1726053853567.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("question_v_statement_finetuned_roberta_basev2_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("question_v_statement_finetuned_roberta_basev2_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|question_v_statement_finetuned_roberta_basev2_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|465.0 MB|

## References

https://huggingface.co/mafwalter/question_v_statement_finetuned_roberta-basev2

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForSequenceClassification