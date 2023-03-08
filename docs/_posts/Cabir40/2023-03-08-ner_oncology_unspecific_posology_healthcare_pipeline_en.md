---
layout: model
title: Pipeline to Extract Cancer Therapies and Posology Information
author: John Snow Labs
name: ner_oncology_unspecific_posology_healthcare_pipeline
date: 2023-03-08
tags: [licensed, clinical, oncology, en, ner, treatment, posology]
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

This pretrained pipeline is built on the top of [ner_oncology_unspecific_posology_healthcare](https://nlp.johnsnowlabs.com/2023/01/11/ner_oncology_unspecific_posology_healthcare_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_oncology_unspecific_posology_healthcare_pipeline_en_4.3.0_3.2_1678269380685.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_oncology_unspecific_posology_healthcare_pipeline_en_4.3.0_3.2_1678269380685.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_oncology_unspecific_posology_healthcare_pipeline", "en", "clinical/models")

text = "
he patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses. She is currently receiving his second cycle of chemotherapy and is in good overall condition.
"

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_oncology_unspecific_posology_healthcare_pipeline", "en", "clinical/models")

val text = "
he patient underwent a regimen consisting of adriamycin (60 mg/m2) and cyclophosphamide (600 mg/m2) over six courses. She is currently receiving his second cycle of chemotherapy and is in good overall condition.
"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | chunks           |   begin |   end | entities             |   confidence |
|---:|:-----------------|--------:|------:|:---------------------|-------------:|
|  0 | adriamycin       |      46 |    55 | Cancer_Therapy       |     0.9999   |
|  1 | 60 mg/m2         |      58 |    65 | Posology_Information |     0.807    |
|  2 | cyclophosphamide |      72 |    87 | Cancer_Therapy       |     0.9998   |
|  3 | 600 mg/m2        |      90 |    98 | Posology_Information |     0.9566   |
|  4 | over six courses |     101 |   116 | Posology_Information |     0.689833 |
|  5 | second cycle     |     150 |   161 | Posology_Information |     0.9906   |
|  6 | chemotherapy     |     166 |   177 | Cancer_Therapy       |     0.9997   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_oncology_unspecific_posology_healthcare_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|533.1 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel