---
layout: model
title: English deberta_v3_base_4bit_64rank_backbone_pipeline pipeline DeBertaForSequenceClassification from yxli2123
author: John Snow Labs
name: deberta_v3_base_4bit_64rank_backbone_pipeline
date: 2024-09-09
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

Pretrained DeBertaForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`deberta_v3_base_4bit_64rank_backbone_pipeline` is a English model originally trained by yxli2123.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/deberta_v3_base_4bit_64rank_backbone_pipeline_en_5.5.0_3.0_1725859611807.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/deberta_v3_base_4bit_64rank_backbone_pipeline_en_5.5.0_3.0_1725859611807.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("deberta_v3_base_4bit_64rank_backbone_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("deberta_v3_base_4bit_64rank_backbone_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|deberta_v3_base_4bit_64rank_backbone_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|689.9 MB|

## References

https://huggingface.co/yxli2123/deberta-v3-base-4bit-64rank-backbone

## Included Models

- DocumentAssembler
- TokenizerModel
- DeBertaForSequenceClassification