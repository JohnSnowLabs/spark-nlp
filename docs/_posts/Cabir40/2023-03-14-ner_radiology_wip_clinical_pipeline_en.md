---
layout: model
title: Pipeline to Detect radiology concepts (ner_radiology_wip_clinical)
author: John Snow Labs
name: ner_radiology_wip_clinical_pipeline
date: 2023-03-14
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

This pretrained pipeline is built on the top of [ner_radiology_wip_clinical](https://nlp.johnsnowlabs.com/2021/04/01/ner_radiology_wip_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_radiology_wip_clinical_pipeline_en_4.3.0_3.2_1678801944623.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_radiology_wip_clinical_pipeline_en_4.3.0_3.2_1678801944623.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_radiology_wip_clinical_pipeline", "en", "clinical/models")

text = '''Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_radiology_wip_clinical_pipeline", "en", "clinical/models")

val text = "Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunks            |   begin |   end | ner_label                 |   confidence |
|---:|:----------------------|--------:|------:|:--------------------------|-------------:|
|  0 | Bilateral             |       0 |     8 | Direction                 |     0.9828   |
|  1 | breast                |      10 |    15 | BodyPart                  |     0.8169   |
|  2 | ultrasound            |      17 |    26 | ImagingTest               |     0.6216   |
|  3 | ovoid mass            |      78 |    87 | ImagingFindings           |     0.6917   |
|  4 | 0.5 x 0.5 x 0.4       |     113 |   127 | Measurements              |     0.91524  |
|  5 | cm                    |     129 |   130 | Units                     |     0.9987   |
|  6 | anteromedial aspect   |     163 |   181 | Direction                 |     0.8241   |
|  7 | left                  |     190 |   193 | Direction                 |     0.4667   |
|  8 | shoulder              |     195 |   202 | BodyPart                  |     0.6349   |
|  9 | mass                  |     210 |   213 | ImagingFindings           |     0.9611   |
| 10 | isoechoic echotexture |     228 |   248 | ImagingFindings           |     0.6851   |
| 11 | muscle                |     266 |   271 | BodyPart                  |     0.7805   |
| 12 | internal color flow   |     294 |   312 | ImagingFindings           |     0.5153   |
| 13 | benign fibrous tissue |     334 |   354 | ImagingFindings           |     0.394867 |
| 14 | lipoma                |     361 |   366 | Disease_Syndrome_Disorder |     0.9142   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_radiology_wip_clinical_pipeline|
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