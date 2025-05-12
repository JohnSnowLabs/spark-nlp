---
layout: model
title: English opensearch_neural_sparse_encoding_doc_v3_distill_pipeline pipeline DistilBertEmbeddings from opensearch-project
author: John Snow Labs
name: opensearch_neural_sparse_encoding_doc_v3_distill_pipeline
date: 2025-04-01
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

Pretrained DistilBertEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`opensearch_neural_sparse_encoding_doc_v3_distill_pipeline` is a English model originally trained by opensearch-project.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/opensearch_neural_sparse_encoding_doc_v3_distill_pipeline_en_5.5.1_3.0_1743523343649.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/opensearch_neural_sparse_encoding_doc_v3_distill_pipeline_en_5.5.1_3.0_1743523343649.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("opensearch_neural_sparse_encoding_doc_v3_distill_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("opensearch_neural_sparse_encoding_doc_v3_distill_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|opensearch_neural_sparse_encoding_doc_v3_distill_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|247.3 MB|

## References

https://huggingface.co/opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill

## Included Models

- DocumentAssembler
- TokenizerModel
- DistilBertEmbeddings