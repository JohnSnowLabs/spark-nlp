---
layout: model
title: English msmarco_v2_bert_mlm_csv30k_pipeline pipeline BertEmbeddings from pxyu
author: John Snow Labs
name: msmarco_v2_bert_mlm_csv30k_pipeline
date: 2025-02-01
tags: [en, open_source, pipeline, onnx]
task: Embeddings
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`msmarco_v2_bert_mlm_csv30k_pipeline` is a English model originally trained by pxyu.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/msmarco_v2_bert_mlm_csv30k_pipeline_en_5.5.1_3.0_1738437727769.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/msmarco_v2_bert_mlm_csv30k_pipeline_en_5.5.1_3.0_1738437727769.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("msmarco_v2_bert_mlm_csv30k_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("msmarco_v2_bert_mlm_csv30k_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|msmarco_v2_bert_mlm_csv30k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|406.7 MB|

## References

https://huggingface.co/pxyu/MSMARCO-V2-BERT-MLM-CSV30k

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings