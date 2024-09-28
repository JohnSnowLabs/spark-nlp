---
layout: model
title: Multilingual cs505_dev_csi_bert_base_multilingual_uncased_pipeline pipeline BertForSequenceClassification from ThuyNT03
author: John Snow Labs
name: cs505_dev_csi_bert_base_multilingual_uncased_pipeline
date: 2024-09-27
tags: [xx, open_source, pipeline, onnx]
task: Text Classification
language: xx
edition: Spark NLP 5.5.0
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained BertForSequenceClassification, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`cs505_dev_csi_bert_base_multilingual_uncased_pipeline` is a Multilingual model originally trained by ThuyNT03.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/cs505_dev_csi_bert_base_multilingual_uncased_pipeline_xx_5.5.0_3.0_1727414058052.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/cs505_dev_csi_bert_base_multilingual_uncased_pipeline_xx_5.5.0_3.0_1727414058052.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("cs505_dev_csi_bert_base_multilingual_uncased_pipeline", lang = "xx")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("cs505_dev_csi_bert_base_multilingual_uncased_pipeline", lang = "xx")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|cs505_dev_csi_bert_base_multilingual_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.0+|
|License:|Open Source|
|Edition:|Official|
|Language:|xx|
|Size:|627.8 MB|

## References

https://huggingface.co/ThuyNT03/CS505-Dev-CSI-bert-base-multilingual-uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- BertForSequenceClassification