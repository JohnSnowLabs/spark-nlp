---
layout: model
title: Pipeline to Extract neurologic deficits related to Stroke Scale (NIHSS)
author: John Snow Labs
name: ner_nihss_pipeline
date: 2023-03-14
tags: [ner, en, licensed, clinical]
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

This pretrained pipeline is built on the top of [ner_nihss](https://nlp.johnsnowlabs.com/2021/11/15/ner_nihss_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_nihss_pipeline_en_4.3.0_3.2_1678778218996.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_nihss_pipeline_en_4.3.0_3.2_1678778218996.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_nihss_pipeline", "en", "clinical/models")

text = '''Abdomen , soft , nontender . NIH stroke scale on presentation was 23 to 24 for , one for consciousness , two for month and year and two for eye / grip , one to two for gaze , two for face , eight for motor , one for limited ataxia , one to two for sensory , three for best language and two for attention . On the neurologic examination the patient was intermittently'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_nihss_pipeline", "en", "clinical/models")

val text = "Abdomen , soft , nontender . NIH stroke scale on presentation was 23 to 24 for , one for consciousness , two for month and year and two for eye / grip , one to two for gaze , two for face , eight for motor , one for limited ataxia , one to two for sensory , three for best language and two for attention . On the neurologic examination the patient was intermittently"

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks       |   begin |   end | ner_label       |   confidence |
|---:|:-----------------|--------:|------:|:----------------|-------------:|
|  0 | NIH stroke scale |      29 |    44 | NIHSS           |     0.973533 |
|  1 | 23 to 24         |      66 |    73 | Measurement     |     0.870567 |
|  2 | one              |      81 |    83 | Measurement     |     0.8726   |
|  3 | consciousness    |      89 |   101 | 1a_LOC          |     0.6322   |
|  4 | two              |     105 |   107 | Measurement     |     0.9665   |
|  5 | month and year   |     113 |   126 | 1b_LOCQuestions |     0.846433 |
|  6 | two              |     132 |   134 | Measurement     |     0.9659   |
|  7 | eye / grip       |     140 |   149 | 1c_LOCCommands  |     0.889433 |
|  8 | one              |     153 |   155 | Measurement     |     0.9917   |
|  9 | two              |     160 |   162 | Measurement     |     0.5144   |
| 10 | gaze             |     168 |   171 | 2_BestGaze      |     0.7272   |
| 11 | two              |     175 |   177 | Measurement     |     0.9872   |
| 12 | face             |     183 |   186 | 4_FacialPalsy   |     0.8758   |
| 13 | eight            |     190 |   194 | Measurement     |     0.9013   |
| 14 | one              |     208 |   210 | Measurement     |     0.9343   |
| 15 | limited          |     216 |   222 | 7_LimbAtaxia    |     0.9326   |
| 16 | ataxia           |     224 |   229 | 7_LimbAtaxia    |     0.5762   |
| 17 | one to two       |     233 |   242 | Measurement     |     0.79     |
| 18 | sensory          |     248 |   254 | 8_Sensory       |     0.9892   |
| 19 | three            |     258 |   262 | Measurement     |     0.8896   |
| 20 | best language    |     268 |   280 | 9_BestLanguage  |     0.89415  |
| 21 | two              |     286 |   288 | Measurement     |     0.949    |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_nihss_pipeline|
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