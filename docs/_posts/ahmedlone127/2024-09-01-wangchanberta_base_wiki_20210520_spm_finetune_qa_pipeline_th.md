---
layout: model
title: Thai wangchanberta_base_wiki_20210520_spm_finetune_qa_pipeline pipeline CamemBertForQuestionAnswering from airesearch
author: John Snow Labs
name: wangchanberta_base_wiki_20210520_spm_finetune_qa_pipeline
date: 2024-09-01
tags: [th, open_source, pipeline, onnx]
task: Question Answering
language: th
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained CamemBertForQuestionAnswering, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`wangchanberta_base_wiki_20210520_spm_finetune_qa_pipeline` is a Thai model originally trained by airesearch.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wangchanberta_base_wiki_20210520_spm_finetune_qa_pipeline_th_5.4.2_3.0_1725162005551.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/wangchanberta_base_wiki_20210520_spm_finetune_qa_pipeline_th_5.4.2_3.0_1725162005551.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("wangchanberta_base_wiki_20210520_spm_finetune_qa_pipeline", lang = "th")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("wangchanberta_base_wiki_20210520_spm_finetune_qa_pipeline", lang = "th")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|wangchanberta_base_wiki_20210520_spm_finetune_qa_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|th|
|Size:|389.4 MB|

## References

https://huggingface.co/airesearch/wangchanberta-base-wiki-20210520-spm-finetune-qa

## Included Models

- MultiDocumentAssembler
- CamemBertForQuestionAnswering