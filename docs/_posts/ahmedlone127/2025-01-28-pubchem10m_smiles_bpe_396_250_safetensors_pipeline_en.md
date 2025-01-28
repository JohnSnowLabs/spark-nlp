---
layout: model
title: English pubchem10m_smiles_bpe_396_250_safetensors_pipeline pipeline RoBertaEmbeddings from yzimmermann
author: John Snow Labs
name: pubchem10m_smiles_bpe_396_250_safetensors_pipeline
date: 2025-01-28
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

Pretrained RoBertaEmbeddings, adapted from Hugging Face and curated to provide scalability and production-readiness using Spark NLP.`pubchem10m_smiles_bpe_396_250_safetensors_pipeline` is a English model originally trained by yzimmermann.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pubchem10m_smiles_bpe_396_250_safetensors_pipeline_en_5.5.1_3.0_1738102803988.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/pubchem10m_smiles_bpe_396_250_safetensors_pipeline_en_5.5.1_3.0_1738102803988.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

pipeline = PretrainedPipeline("pubchem10m_smiles_bpe_396_250_safetensors_pipeline", lang = "en")
annotations =  pipeline.transform(df)   

```
```scala

val pipeline = new PretrainedPipeline("pubchem10m_smiles_bpe_396_250_safetensors_pipeline", lang = "en")
val annotations = pipeline.transform(df)

```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pubchem10m_smiles_bpe_396_250_safetensors_pipeline|
|Type:|pipeline|
|Compatibility:|Spark NLP 5.5.1+|
|License:|Open Source|
|Edition:|Official|
|Language:|en|
|Size:|311.0 MB|

## References

https://huggingface.co/yzimmermann/PubChem10M_SMILES_BPE_396_250-safetensors

## Included Models

- DocumentAssembler
- TokenizerModel
- RoBertaEmbeddings