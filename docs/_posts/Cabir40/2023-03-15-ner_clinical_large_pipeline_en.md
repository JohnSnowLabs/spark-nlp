---
layout: model
title: Pipeline to Detect Problems, Tests and Treatments (ner_clinical_large)
author: John Snow Labs
name: ner_clinical_large_pipeline
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

This pretrained pipeline is built on the top of [ner_clinical_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_clinical_large_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_clinical_large_pipeline_en_4.3.0_3.2_1678876271920.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_clinical_large_pipeline_en_4.3.0_3.2_1678876271920.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_clinical_large_pipeline", "en", "clinical/models")

text = '''The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes. BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_clinical_large_pipeline", "en", "clinical/models")

val text = "The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes. BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                                                   |   begin |   end | ner_label   |   confidence |
|---:|:------------------------------------------------------------|--------:|------:|:------------|-------------:|
|  0 | the G-protein-activated inwardly rectifying potassium (GIRK |      48 |   106 | TREATMENT   |     0.6926   |
|  1 | the genomicorganization                                     |     142 |   164 | TREATMENT   |     0.80715  |
|  2 | a candidate gene forType II diabetes mellitus               |     210 |   254 | PROBLEM     |     0.754343 |
|  3 | byapproximately                                             |     380 |   394 | TREATMENT   |     0.7924   |
|  4 | single nucleotide polymorphisms                             |     464 |   494 | TREATMENT   |     0.636967 |
|  5 | aVal366Ala substitution                                     |     532 |   554 | PROBLEM     |     0.53615  |
|  6 | an 8 base-pair                                              |     561 |   574 | PROBLEM     |     0.607733 |
|  7 | insertion/deletion                                          |     581 |   598 | PROBLEM     |     0.8692   |
|  8 | Ourexpression studies                                       |     601 |   621 | TEST        |     0.89975  |
|  9 | the transcript in various humantissues                      |     648 |   685 | PROBLEM     |     0.83306  |
| 10 | fat andskeletal muscle                                      |     749 |   770 | PROBLEM     |     0.778133 |
| 11 | furtherstudies                                              |     830 |   843 | PROBLEM     |     0.8789   |
| 12 | the KCNJ9 protein                                           |     864 |   880 | TREATMENT   |     0.561033 |
| 13 | evaluation                                                  |     892 |   901 | TEST        |     0.9981   |
| 14 | Type II diabetes                                            |     940 |   955 | PROBLEM     |     0.698967 |
| 15 | the treatment                                               |    1025 |  1037 | TREATMENT   |     0.81195  |
| 16 | breast cancer                                               |    1042 |  1054 | PROBLEM     |     0.9604   |
| 17 | the standard therapy                                        |    1067 |  1086 | TREATMENT   |     0.757767 |
| 18 | anthracyclines                                              |    1125 |  1138 | TREATMENT   |     0.9999   |
| 19 | taxanes                                                     |    1144 |  1150 | TREATMENT   |     0.9999   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_clinical_large_pipeline|
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