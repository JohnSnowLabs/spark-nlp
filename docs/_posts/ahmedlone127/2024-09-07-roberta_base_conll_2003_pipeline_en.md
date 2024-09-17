---
layout: model
title: English roberta_base_conll_2003_pipeline pipeline RoBertaForTokenClassification from CheccoCando
author: John Snow Labs
name: roberta_base_conll_2003_pipeline
date: 2024-09-07
tags: [en, open_source, pipeline, onnx]
task: Named Entity Recognition
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

Pretrained RoBertaForTokenClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`roberta_base_conll_2003_pipeline` is a English model originally trained by CheccoCando.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/roberta_base_conll_2003_pipeline_en_5.5.0_3.0_1725723641775.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/roberta_base_conll_2003_pipeline_en_5.5.0_3.0_1725723641775.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("roberta_base_conll_2003_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("roberta_base_conll_2003_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|roberta_base_conll_2003_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|441.6 MB|

## References

https://huggingface.co/CheccoCando/roberta-base_CoNLL_2003

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaForTokenClassification