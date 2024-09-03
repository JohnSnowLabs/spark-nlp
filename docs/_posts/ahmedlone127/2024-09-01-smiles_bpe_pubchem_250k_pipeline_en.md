---
layout: model
title: English smiles_bpe_pubchem_250k_pipeline pipeline RoBertaEmbeddings from seyonec
author: John Snow Labs
name: smiles_bpe_pubchem_250k_pipeline
date: 2024-09-01
tags: [en, open_source, pipeline, onnx]
task: Embeddings
language: en
edition: Spark NLP 5.4.2
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`smiles_bpe_pubchem_250k_pipeline` is a English model originally trained by seyonec.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/smiles_bpe_pubchem_250k_pipeline_en_5.4.2_3.0_1725191518185.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/smiles_bpe_pubchem_250k_pipeline_en_5.4.2_3.0_1725191518185.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("smiles_bpe_pubchem_250k_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("smiles_bpe_pubchem_250k_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|smiles_bpe_pubchem_250k_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.4.2+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|310.0 MB|

## References

https://huggingface.co/seyonec/SMILES_BPE_PubChem_250k

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings