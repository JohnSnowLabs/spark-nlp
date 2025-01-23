---
layout: model
title: Latin sent_xlm_roberta_base_latin_uncased_pipeline pipeline XlmRoBertaSentenceEmbeddings from Cicciokr
author: John Snow Labs
name: sent_xlm_roberta_base_latin_uncased_pipeline
date: 2025-01-23
tags: [la, open_source, pipeline, onnx]
task: Embeddings
language: la
edition: Spark NLP 5.5.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained XlmRoBertaSentenceEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`sent_xlm_roberta_base_latin_uncased_pipeline` is a Latin model originally trained by Cicciokr.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/sent_xlm_roberta_base_latin_uncased_pipeline_la_5.5.1_3.0_1737661880094.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/sent_xlm_roberta_base_latin_uncased_pipeline_la_5.5.1_3.0_1737661880094.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("sent_xlm_roberta_base_latin_uncased_pipeline", lang = "la")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("sent_xlm_roberta_base_latin_uncased_pipeline", lang = "la")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|sent_xlm_roberta_base_latin_uncased_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|la|
|Size:|916.7 MB|

## References

https://huggingface.co/Cicciokr/XLM-Roberta-Base-Latin-Uncased

## Included Models

- DocumentAssembler
- TokenizerModel
- SentenceDetectorDLModel
- XlmRoBertaSentenceEmbeddings