---
layout: model
title: Pipeline to Extract the Names of Drugs & Chemicals
author: John Snow Labs
name: ner_chemd_clinical_pipeline
date: 2023-03-14
tags: [chemdner, chemd, ner, clinical, en, licensed]
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

This pretrained pipeline is built on the top of [ner_chemd_clinical](https://nlp.johnsnowlabs.com/2021/11/04/ner_chemd_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_chemd_clinical_pipeline_en_4.3.0_3.2_1678778578175.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_chemd_clinical_pipeline_en_4.3.0_3.2_1678778578175.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_chemd_clinical_pipeline", "en", "clinical/models")

text = '''Isolation, Structure Elucidation, and Iron-Binding Properties of Lystabactins, Siderophores Isolated from a Marine Pseudoalteromonas sp. The marine bacterium Pseudoalteromonas sp. S2B, isolated from the Gulf of Mexico after the Deepwater Horizon oil spill, was found to produce lystabactins A, B, and C (1-3), three new siderophores. The structures were elucidated through mass spectrometry, amino acid analysis, and NMR. The lystabactins are composed of serine (Ser), asparagine (Asn), two formylated/hydroxylated ornithines (FOHOrn), dihydroxy benzoic acid (Dhb), and a very unusual nonproteinogenic amino acid, 4,8-diamino-3-hydroxyoctanoic acid (LySta). The iron-binding properties of the compounds were investigated through a spectrophotometric competition.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_chemd_clinical_pipeline", "en", "clinical/models")

val text = "Isolation, Structure Elucidation, and Iron-Binding Properties of Lystabactins, Siderophores Isolated from a Marine Pseudoalteromonas sp. The marine bacterium Pseudoalteromonas sp. S2B, isolated from the Gulf of Mexico after the Deepwater Horizon oil spill, was found to produce lystabactins A, B, and C (1-3), three new siderophores. The structures were elucidated through mass spectrometry, amino acid analysis, and NMR. The lystabactins are composed of serine (Ser), asparagine (Asn), two formylated/hydroxylated ornithines (FOHOrn), dihydroxy benzoic acid (Dhb), and a very unusual nonproteinogenic amino acid, 4,8-diamino-3-hydroxyoctanoic acid (LySta). The iron-binding properties of the compounds were investigated through a spectrophotometric competition."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks                         |   begin |   end | ner_label    |   confidence |
|---:|:-----------------------------------|--------:|------:|:-------------|-------------:|
|  0 | Lystabactins                       |      65 |    76 | FAMILY       |     0.9841   |
|  1 | lystabactins A, B, and C           |     278 |   301 | MULTIPLE     |     0.813429 |
|  2 | amino acid                         |     392 |   401 | FAMILY       |     0.74585  |
|  3 | lystabactins                       |     426 |   437 | FAMILY       |     0.8007   |
|  4 | serine                             |     455 |   460 | TRIVIAL      |     0.9924   |
|  5 | Ser                                |     463 |   465 | FORMULA      |     0.9999   |
|  6 | asparagine                         |     469 |   478 | TRIVIAL      |     0.9795   |
|  7 | Asn                                |     481 |   483 | FORMULA      |     0.9999   |
|  8 | formylated/hydroxylated ornithines |     491 |   524 | FAMILY       |     0.50085  |
|  9 | FOHOrn                             |     527 |   532 | FORMULA      |     0.509    |
| 10 | dihydroxy benzoic acid             |     536 |   557 | SYSTEMATIC   |     0.6346   |
| 11 | amino acid                         |     602 |   611 | FAMILY       |     0.4204   |
| 12 | 4,8-diamino-3-hydroxyoctanoic acid |     614 |   647 | SYSTEMATIC   |     0.9124   |
| 13 | LySta                              |     650 |   654 | ABBREVIATION |     0.9193   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_chemd_clinical_pipeline|
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