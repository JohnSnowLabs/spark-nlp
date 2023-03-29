---
layout: model
title: Pipeline to Detect Genes/Proteins (BC2GM) in Medical Text
author: John Snow Labs
name: bert_token_classifier_ner_bc2gm_gene_pipeline
date: 2023-03-20
tags: [en, ner, clinical, licensed, bertfortokenclassification]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 4.3.0
spark_version: 3.2
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [bert_token_classifier_ner_bc2gm_gene](https://nlp.johnsnowlabs.com/2022/07/25/bert_token_classifier_ner_bc2gm_gene_en_3_0.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc2gm_gene_pipeline_en_4.3.0_3.2_1679303903870.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_bc2gm_gene_pipeline_en_4.3.0_3.2_1679303903870.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_bc2gm_gene_pipeline", "en", "clinical/models")

text = '''ROCK-I, Kinectin, and mDia2 can bind the wild type forms of both RhoA and Cdc42 in a GTP-dependent manner in vitro. These results support the hypothesis that in the presence of tryptophan the ribosome translating tnaC blocks Rho ' s access to the boxA and rut sites, thereby preventing transcription termination.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_bc2gm_gene_pipeline", "en", "clinical/models")

val text = "ROCK-I, Kinectin, and mDia2 can bind the wild type forms of both RhoA and Cdc42 in a GTP-dependent manner in vitro. These results support the hypothesis that in the presence of tryptophan the ribosome translating tnaC blocks Rho ' s access to the boxA and rut sites, thereby preventing transcription termination."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk   |   begin |   end | ner_label    |   confidence |
|---:|:------------|--------:|------:|:-------------|-------------:|
|  0 | ROCK-I      |       0 |     5 | GENE/PROTEIN |     0.999978 |
|  1 | Kinectin    |       8 |    15 | GENE/PROTEIN |     0.999973 |
|  2 | mDia2       |      22 |    26 | GENE/PROTEIN |     0.999974 |
|  3 | RhoA        |      65 |    68 | GENE/PROTEIN |     0.999976 |
|  4 | Cdc42       |      74 |    78 | GENE/PROTEIN |     0.999979 |
|  5 | tnaC        |     213 |   216 | GENE/PROTEIN |     0.999978 |
|  6 | Rho         |     225 |   227 | GENE/PROTEIN |     0.999976 |
|  7 | boxA        |     247 |   250 | GENE/PROTEIN |     0.999837 |
|  8 | rut sites   |     256 |   264 | GENE/PROTEIN |     0.99115  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_bc2gm_gene_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.7 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel