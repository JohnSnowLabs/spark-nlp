---
layout: model
title: Pipeline to Detect Normalized Genes and Human Phenotypes
author: John Snow Labs
name: ner_human_phenotype_gene_clinical_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, gene, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [ner_human_phenotype_gene_clinical](https://nlp.johnsnowlabs.com/2021/03/31/ner_human_phenotype_gene_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_clinical_pipeline_en_3.4.1_3.0_1647867667569.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_human_phenotype_gene_clinical_pipeline_en_3.4.1_3.0_1647867667569.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_human_phenotype_gene_clinical_pipeline", "en", "clinical/models")

pipeline.annotate("Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3).")
```
```scala
val pipeline = new PretrainedPipeline("ner_human_phenotype_gene_clinical_pipeline", "en", "clinical/models")

pipeline.annotate("Here we presented a case (BS type) of a 17 years old female presented with polyhydramnios, polyuria, nephrocalcinosis and hypokalemia, which was alleviated after treatment with celecoxib and vitamin D(3).")
```
</div>

## Results

```bash
+----+------------------+---------+-------+----------+
|    | chunk            |   begin |   end | entity   |
+====+==================+=========+=======+==========+
|  0 | BS type          |      29 |    32 | GENE     |
+----+------------------+---------+-------+----------+
|  1 | polyhydramnios   |      75 |    88 | HP       |
+----+------------------+---------+-------+----------+
|  2 | polyuria         |      91 |    98 | HP       |
+----+------------------+---------+-------+----------+
|  3 | nephrocalcinosis |     101 |   116 | HP       |
+----+------------------+---------+-------+----------+
|  4 | hypokalemia      |     122 |   132 | HP       |
+----+------------------+---------+-------+----------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_human_phenotype_gene_clinical_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
