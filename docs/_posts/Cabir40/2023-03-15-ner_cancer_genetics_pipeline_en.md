---
layout: model
title: Pipeline to Detect Genetic Cancer Entities
author: John Snow Labs
name: ner_cancer_genetics_pipeline
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

This pretrained pipeline is built on the top of [ner_cancer_genetics](https://nlp.johnsnowlabs.com/2021/03/31/ner_cancer_genetics_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_cancer_genetics_pipeline_en_4.3.0_3.2_1678864026558.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_cancer_genetics_pipeline_en_4.3.0_3.2_1678864026558.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_cancer_genetics_pipeline", "en", "clinical/models")

text = '''The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_cancer_genetics_pipeline", "en", "clinical/models")

val text = "The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                                                               |   begin |   end | ner_label   |   confidence |
|---:|:------------------------------------------------------------------------|--------:|------:|:------------|-------------:|
|  0 | human KCNJ9                                                             |       4 |    14 | protein     |     0.674    |
|  1 | Kir 3.3                                                                 |      17 |    23 | protein     |     0.95355  |
|  2 | GIRK3                                                                   |      26 |    30 | protein     |     0.5127   |
|  3 | G-protein-activated inwardly rectifying potassium (GIRK) channel family |      52 |   122 | protein     |     0.691744 |
|  4 | KCNJ9 locus                                                             |     173 |   183 | DNA         |     0.97875  |
|  5 | chromosome 1q21-23                                                      |     188 |   205 | DNA         |     0.95305  |
|  6 | coding exons                                                            |     357 |   368 | DNA         |     0.63345  |
|  7 | identified14 single nucleotide polymorphisms                            |     451 |   494 | DNA         |     0.6994   |
|  8 | SNPs),                                                                  |     497 |   502 | DNA         |     0.79075  |
|  9 | KCNJ9 gene                                                              |     801 |   810 | DNA         |     0.95605  |
| 10 | KCNJ9 protein                                                           |     868 |   880 | protein     |     0.844    |
| 11 | locus                                                                   |     931 |   935 | DNA         |     0.9685   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_cancer_genetics_pipeline|
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