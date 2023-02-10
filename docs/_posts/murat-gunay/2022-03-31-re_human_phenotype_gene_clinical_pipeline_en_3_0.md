---
layout: model
title: Pipeline to Detect Relations Between Genes and Phenotypes
author: John Snow Labs
name: re_human_phenotype_gene_clinical_pipeline
date: 2022-03-31
tags: [licensed, clinical, re, genes, phenotypes, en]
task: Relation Extraction
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

This pretrained pipeline is built on the top of [re_human_phenotype_gene_clinical](https://nlp.johnsnowlabs.com/2020/09/30/re_human_phenotype_gene_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_human_phenotype_gene_clinical_pipeline_en_3.4.1_3.0_1648734276384.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/re_human_phenotype_gene_clinical_pipeline_en_3.4.1_3.0_1648734276384.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("re_human_phenotype_gene_clinical_pipeline", "en", "clinical/models")


pipeline.annotate("Bilateral colobomatous microphthalmia and developmental delay in whom genetic studies identified a homozygous TENM3")
```
```scala
val pipeline = new PretrainedPipeline("re_human_phenotype_gene_clinical_pipeline", "en", "clinical/models")


pipeline.annotate("Bilateral colobomatous microphthalmia and developmental delay in whom genetic studies identified a homozygous TENM3")
```
</div>

## Results

```bash
+----+------------+-----------+-----------------+---------------+---------------------+-----------+-----------------+---------------+---------------------+--------------+
|    |   relation | entity1   |   entity1_begin |   entity1_end | chunk1              | entity2   |   entity2_begin |   entity2_end | chunk2              |   confidence |
+====+============+===========+=================+===============+=====================+===========+=================+===============+=====================+==============+
|  0 |          1 | HP        |              23 |            36 | microphthalmia      | HP        |              42 |            60 | developmental delay |     0.999954 |
+----+------------+-----------+-----------------+---------------+---------------------+-----------+-----------------+---------------+---------------------+--------------+
|  1 |          1 | HP        |              23 |            36 | microphthalmia      | GENE      |             110 |           114 | TENM3               |     0.999999 |
+----+------------+-----------+-----------------+---------------+---------------------+-----------+-----------------+---------------+---------------------+--------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_human_phenotype_gene_clinical_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|1.7 GB|

## Included Models

- DocumentAssembler
- SentenceDetector
- TokenizerModel
- PerceptronModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverter
- DependencyParserModel
- RelationExtractionModel