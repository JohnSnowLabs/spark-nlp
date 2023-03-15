---
layout: model
title: Pipeline to Detect Posology concepts (ner_posology_healthcare)
author: John Snow Labs
name: ner_posology_healthcare_pipeline
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

This pretrained pipeline is built on the top of [ner_posology_healthcare](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_healthcare_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_healthcare_pipeline_en_4.3.0_3.2_1678870448348.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_posology_healthcare_pipeline_en_4.3.0_3.2_1678870448348.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_posology_healthcare_pipeline", "en", "clinical/models")

text = '''The patient is a 40-year-old white male who presents with a chief complaint of 'chest pain'. The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that chest pain started yesterday evening. He has been advised Aspirin 81 milligrams QDay. insulin 50 units in a.m. HCTZ 50 mg QDay. Nitroglycerin 1/150 sublingually.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_posology_healthcare_pipeline", "en", "clinical/models")

val text = "The patient is a 40-year-old white male who presents with a chief complaint of 'chest pain'. The patient is diabetic and has a prior history of coronary artery disease. The patient presents today stating that chest pain started yesterday evening. He has been advised Aspirin 81 milligrams QDay. insulin 50 units in a.m. HCTZ 50 mg QDay. Nitroglycerin 1/150 sublingually."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk     |   begin |   end | ner_label   |   confidence |
|---:|:--------------|--------:|------:|:------------|-------------:|
|  0 | Aspirin       |     267 |   273 | Drug        |      0.9983  |
|  1 | 81 milligrams |     275 |   287 | Strength    |      0.9921  |
|  2 | QDay          |     289 |   292 | Frequency   |      0.995   |
|  3 | insulin       |     295 |   301 | Drug        |      0.9469  |
|  4 | 50 units      |     303 |   310 | Dosage      |      0.6343  |
|  5 | in a.m        |     312 |   317 | Frequency   |      0.71315 |
|  6 | HCTZ          |     320 |   323 | Drug        |      0.9789  |
|  7 | 50 mg         |     325 |   329 | Strength    |      0.96705 |
|  8 | QDay          |     331 |   334 | Frequency   |      0.9877  |
|  9 | Nitroglycerin |     337 |   349 | Drug        |      0.9927  |
| 10 | 1/150         |     351 |   355 | Strength    |      0.9565  |
| 11 | sublingually. |     357 |   369 | Route       |      0.72065 |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_posology_healthcare_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|513.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- WordEmbeddingsModel
- MedicalNerModel
- NerConverterInternalModel