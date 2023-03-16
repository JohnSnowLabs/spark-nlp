---
layout: model
title: Pipeline to Detect Drugs and posology entities including experimental drugs and cycles (ner_posology_experimental)
author: John Snow Labs
name: ner_posology_experimental_pipeline
date: 2023-03-15
tags: [licensed, clinical, en, ner]
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

This pretrained pipeline is built on the top of [ner_posology_experimental](https://nlp.johnsnowlabs.com/2021/09/01/ner_posology_experimental_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_experimental_pipeline_en_4.3.0_3.2_1678870276632.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_posology_experimental_pipeline_en_4.3.0_3.2_1678870276632.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_posology_experimental_pipeline", "en", "clinical/models")

text = '''Y-90 Humanized Anti-Tac: 10 mCi (if a bone marrow transplant was part of the patient's previous therapy) or 15 mCi of yttrium labeled anti-TAC; followed by calcium trisodium Inj (Ca DTPA)..

Calcium-DTPA: Ca-DTPA will be administered intravenously on Days 1-3 to clear the radioactive agent from the body.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_posology_experimental_pipeline", "en", "clinical/models")

val text = "Y-90 Humanized Anti-Tac: 10 mCi (if a bone marrow transplant was part of the patient's previous therapy) or 15 mCi of yttrium labeled anti-TAC; followed by calcium trisodium Inj (Ca DTPA)..

Calcium-DTPA: Ca-DTPA will be administered intravenously on Days 1-3 to clear the radioactive agent from the body."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                |   begin |   end | ner_label   |   confidence |
|---:|:-------------------------|--------:|------:|:------------|-------------:|
|  0 | Anti-Tac                 |      15 |    22 | Drug        |     0.8797   |
|  1 | 10 mCi                   |      25 |    30 | Dosage      |     0.5403   |
|  2 | 15 mCi                   |     108 |   113 | Dosage      |     0.6266   |
|  3 | yttrium labeled anti-TAC |     118 |   141 | Drug        |     0.9122   |
|  4 | calcium trisodium Inj    |     156 |   176 | Drug        |     0.397533 |
|  5 | Calcium-DTPA             |     191 |   202 | Drug        |     0.9794   |
|  6 | Ca-DTPA                  |     205 |   211 | Drug        |     0.9544   |
|  7 | intravenously            |     234 |   246 | Route       |     0.9518   |
|  8 | Days 1-3                 |     251 |   258 | Cycleday    |     0.83325  |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_posology_experimental_pipeline|
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