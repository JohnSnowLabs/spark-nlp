---
layout: model
title: English bmu_finetuned_pubmedbert_mlm_medquad_pipeline pipeline BertEmbeddings from Deepanshu7284
author: John Snow Labs
name: bmu_finetuned_pubmedbert_mlm_medquad_pipeline
date: 2025-04-08
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

Pretrained BertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`bmu_finetuned_pubmedbert_mlm_medquad_pipeline` is a English model originally trained by Deepanshu7284.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bmu_finetuned_pubmedbert_mlm_medquad_pipeline_en_5.5.1_3.0_1744123977482.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bmu_finetuned_pubmedbert_mlm_medquad_pipeline_en_5.5.1_3.0_1744123977482.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("bmu_finetuned_pubmedbert_mlm_medquad_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("bmu_finetuned_pubmedbert_mlm_medquad_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bmu_finetuned_pubmedbert_mlm_medquad_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|408.2 MB|

## References

https://huggingface.co/Deepanshu7284/BMU_Finetuned_PubMedBERT_MLM_MedQUAD

## Included Models

- DocumentAssembler
- TokenizerModel
- BertEmbeddings