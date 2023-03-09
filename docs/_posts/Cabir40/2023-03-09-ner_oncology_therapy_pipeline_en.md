---
layout: model
title: Pipeline to Detect Entities Related to Cancer Therapies
author: John Snow Labs
name: ner_oncology_therapy_pipeline
date: 2023-03-09
tags: [clinical, en, licensed, oncology, treatment, ner]
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

This pretrained pipeline is built on the top of [ner_oncology_therapy](https://nlp.johnsnowlabs.com/2022/11/24/ner_oncology_therapy_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_therapy_pipeline_en_4.3.0_3.2_1678351787302.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_therapy_pipeline_en_4.3.0_3.2_1678351787302.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_oncology_therapy_pipeline", "en", "clinical/models")

text = '''The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago.
The tumor was positive for ER and PR. Postoperatively, radiotherapy was administered to her breast.
The cancer recurred as a right lung metastasis 13 years later. The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses, as first line therapy.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_oncology_therapy_pipeline", "en", "clinical/models")

val text = "The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago.
The tumor was positive for ER and PR. Postoperatively, radiotherapy was administered to her breast.
The cancer recurred as a right lung metastasis 13 years later. The patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses, as first line therapy."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ber_chunks                     |   begin |   end | ner_label             |   confidence |
|---:|:-------------------------------|--------:|------:|:----------------------|-------------:|
|  0 | mastectomy                     |      36 |    45 | Cancer_Surgery        |     0.9817   |
|  1 | axillary lymph node dissection |      54 |    83 | Cancer_Surgery        |     0.719725 |
|  2 | radiotherapy                   |     183 |   194 | Radiotherapy          |     0.9984   |
|  3 | recurred                       |     239 |   246 | Response_To_Treatment |     0.9481   |
|  4 | adriamycin                     |     337 |   346 | Chemotherapy          |     0.9981   |
|  5 | 60 mg/m2                       |     349 |   356 | Dosage                |     0.58815  |
|  6 | cyclophosphamide               |     363 |   378 | Chemotherapy          |     0.9976   |
|  7 | 600 mg/m2                      |     381 |   389 | Dosage                |     0.64205  |
|  8 | six courses                    |     397 |   407 | Cycle_Count           |     0.46815  |
|  9 | first line                     |     413 |   422 | Line_Of_Therapy       |     0.95015  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_therapy_pipeline|
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