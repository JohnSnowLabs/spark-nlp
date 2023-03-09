---
layout: model
title: Pipeline to Detect Oncology-Specific Entities
author: John Snow Labs
name: ner_oncology_pipeline
date: 2023-03-08
tags: [licensed, clinical, en, oncology, biomarker, treatment]
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

This pretrained pipeline is built on the top of [ner_oncology](https://nlp.johnsnowlabs.com/2022/11/24/ner_oncology_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_pipeline_en_4.3.0_3.2_1678283681531.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_pipeline_en_4.3.0_3.2_1678283681531.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_oncology_pipeline", "en", "clinical/models")

text = '''The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago.
The tumor was positive for ER and PR. Postoperatively, radiotherapy was administered to the residual breast.
The cancer recurred as a right lung metastasis 13 years later. The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses, as first line therapy.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_oncology_pipeline", "en", "clinical/models")

val text = "The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago.
The tumor was positive for ER and PR. Postoperatively, radiotherapy was administered to the residual breast.
The cancer recurred as a right lung metastasis 13 years later. The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses, as first line therapy."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ber_chunks                     |   begin |   end | ner_label             |   confidence |
|---:|:-------------------------------|--------:|------:|:----------------------|-------------:|
|  0 | left                           |      31 |    34 | Direction             |     0.9913   |
|  1 | mastectomy                     |      36 |    45 | Cancer_Surgery        |     0.952    |
|  2 | axillary lymph node dissection |      54 |    83 | Cancer_Surgery        |     0.744525 |
|  3 | left                           |      91 |    94 | Direction             |     0.9966   |
|  4 | breast cancer                  |      96 |   108 | Cancer_Dx             |     0.9272   |
|  5 | twenty years ago               |     110 |   125 | Relative_Date         |     0.857067 |
|  6 | tumor                          |     132 |   136 | Tumor_Finding         |     0.9959   |
|  7 | positive                       |     142 |   149 | Biomarker_Result      |     0.9958   |
|  8 | ER                             |     155 |   156 | Biomarker             |     0.9952   |
|  9 | PR                             |     162 |   163 | Biomarker             |     0.9709   |
| 10 | radiotherapy                   |     183 |   194 | Radiotherapy          |     0.9997   |
| 11 | breast                         |     229 |   234 | Site_Breast           |     0.8288   |
| 12 | cancer                         |     241 |   246 | Cancer_Dx             |     0.9949   |
| 13 | recurred                       |     248 |   255 | Response_To_Treatment |     0.9849   |
| 14 | right                          |     262 |   266 | Direction             |     0.9993   |
| 15 | lung                           |     268 |   271 | Site_Lung             |     0.9982   |
| 16 | metastasis                     |     273 |   282 | Metastasis            |     0.9999   |
| 17 | 13 years later                 |     284 |   297 | Relative_Date         |     0.791433 |
| 18 | adriamycin                     |     346 |   355 | Chemotherapy          |     0.9999   |
| 19 | 60 mg/m2                       |     358 |   365 | Dosage                |     0.91785  |
| 20 | cyclophosphamide               |     372 |   387 | Chemotherapy          |     0.9999   |
| 21 | 600 mg/m2                      |     390 |   398 | Dosage                |     0.9647   |
| 22 | six courses                    |     406 |   416 | Cycle_Count           |     0.6798   |
| 23 | first line                     |     422 |   431 | Line_Of_Therapy       |     0.9792   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_pipeline|
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