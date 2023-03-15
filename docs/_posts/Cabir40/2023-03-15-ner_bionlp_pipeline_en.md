---
layout: model
title: Pipeline to Detect Cancer Genetics
author: John Snow Labs
name: ner_bionlp_pipeline
date: 2023-03-15
tags: [ner, clinical, licensed, en]
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

This pretrained pipeline is built on the top of [ner_bionlp](https://nlp.johnsnowlabs.com/2021/03/31/ner_bionlp_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_pipeline_en_4.3.0_3.2_1678865044035.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_bionlp_pipeline_en_4.3.0_3.2_1678865044035.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_bionlp_pipeline", "en", "clinical/models")

text = '''The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_bionlp_pipeline", "en", "clinical/models")

val text = "The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk              |   begin |   end | ner_label            |   confidence |
|---:|:-----------------------|--------:|------:|:---------------------|-------------:|
|  0 | human                  |       4 |     8 | Organism             |     0.9996   |
|  1 | Kir 3.3                |      17 |    23 | Gene_or_gene_product |     0.99635  |
|  2 | GIRK3                  |      26 |    30 | Gene_or_gene_product |     1        |
|  3 | potassium              |      92 |   100 | Simple_chemical      |     0.9452   |
|  4 | GIRK                   |     103 |   106 | Gene_or_gene_product |     0.998    |
|  5 | chromosome 1q21-23     |     188 |   205 | Cellular_component   |     0.80115  |
|  6 | pancreas               |     697 |   704 | Organ                |     0.9994   |
|  7 | tissues                |     740 |   746 | Tissue               |     0.975    |
|  8 | fat andskeletal muscle |     749 |   770 | Tissue               |     0.955433 |
|  9 | KCNJ9                  |     801 |   805 | Gene_or_gene_product |     0.9172   |
| 10 | Type II                |     940 |   946 | Gene_or_gene_product |     0.98845  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_bionlp_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
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
- NerConverterInternalModel