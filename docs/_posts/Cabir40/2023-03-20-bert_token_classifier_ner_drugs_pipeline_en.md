---
layout: model
title: Pipeline to Detect Drug Chemicals (BertForTokenClassifier)
author: John Snow Labs
name: bert_token_classifier_ner_drugs_pipeline
date: 2023-03-20
tags: [drug, berfortokenclassification, ner, en, licensed]
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

This pretrained pipeline is built on the top of [bert_token_classifier_ner_drugs](https://nlp.johnsnowlabs.com/2022/01/06/bert_token_classifier_ner_drugs_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_drugs_pipeline_en_4.3.0_3.2_1679307572006.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/bert_token_classifier_ner_drugs_pipeline_en_4.3.0_3.2_1679307572006.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("bert_token_classifier_ner_drugs_pipeline", "en", "clinical/models")

text = '''The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes. With the objective of determining the usefulnessof vinorelbine monotherapy in patients with advanced or recurrent breast cancerafter standard therapy, we evaluated the efficacy and safety of vinorelbine inpatients previously treated with anthracyclines and taxanes.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("bert_token_classifier_ner_drugs_pipeline", "en", "clinical/models")

val text = "The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family. Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II diabetes mellitus in the Pima Indian population. The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons separated byapproximately 2.2 and approximately 2.6 kb introns, respectively. We identified14 single nucleotide polymorphisms (SNPs), including one that predicts aVal366Ala substitution, and an 8 base-pair (bp) insertion/deletion. Ourexpression studies revealed the presence of the transcript in various humantissues including pancreas, and two major insulin-responsive tissues: fat andskeletal muscle. The characterization of the KCNJ9 gene should facilitate furtherstudies on the function of the KCNJ9 protein and allow evaluation of thepotential role of the locus in Type II diabetes.BACKGROUND: At present, it is one of the most important issues for the treatment of breast cancer to develop the standard therapy for patients previously treated with anthracyclines and taxanes. With the objective of determining the usefulnessof vinorelbine monotherapy in patients with advanced or recurrent breast cancerafter standard therapy, we evaluated the efficacy and safety of vinorelbine inpatients previously treated with anthracyclines and taxanes."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk      |   begin |   end | ner_label   |   confidence |
|---:|:---------------|--------:|------:|:------------|-------------:|
|  0 | potassium      |      92 |   100 | DrugChem    |     0.990254 |
|  1 | nucleotide     |     471 |   480 | DrugChem    |     0.500501 |
|  2 | anthracyclines |    1124 |  1137 | DrugChem    |     0.999987 |
|  3 | taxanes        |    1143 |  1149 | DrugChem    |     0.999972 |
|  4 | vinorelbine    |    1203 |  1213 | DrugChem    |     0.999991 |
|  5 | vinorelbine    |    1343 |  1353 | DrugChem    |     0.999991 |
|  6 | anthracyclines |    1390 |  1403 | DrugChem    |     0.99999  |
|  7 | taxanes        |    1409 |  1415 | DrugChem    |     0.999946 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_ner_drugs_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|404.9 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- MedicalBertForTokenClassifier
- NerConverterInternalModel