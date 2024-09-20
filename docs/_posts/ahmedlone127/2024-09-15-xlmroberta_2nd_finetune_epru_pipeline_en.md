---
layout: model
title: English xlmroberta_2nd_finetune_epru_pipeline pipeline XlmRoBertaForSequenceClassification from mmillet
author: John Snow Labs
name: xlmroberta_2nd_finetune_epru_pipeline
date: 2024-09-15
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

Pretrained XlmRoBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`xlmroberta_2nd_finetune_epru_pipeline` is a English model originally trained by mmillet.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlmroberta_2nd_finetune_epru_pipeline_en_5.5.0_3.0_1726434377575.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/xlmroberta_2nd_finetune_epru_pipeline_en_5.5.0_3.0_1726434377575.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("xlmroberta_2nd_finetune_epru_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("xlmroberta_2nd_finetune_epru_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlmroberta_2nd_finetune_epru_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|847.9 MB|

## References

https://huggingface.co/mmillet/xlmroberta-2nd-finetune-epru

## Included Models

- DocumentAssembler
- TokenizerModel
- XlmRoBertaForSequenceClassification