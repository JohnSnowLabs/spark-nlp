---
layout: model
title: Pipeline to Detect Radiology Related Entities
author: John Snow Labs
name: ner_radiology_pipeline
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

This pretrained pipeline is built on the top of [ner_radiology](https://nlp.johnsnowlabs.com/2021/03/31/ner_radiology_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_radiology_pipeline_en_4.3.0_3.2_1678865918152.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_radiology_pipeline_en_4.3.0_3.2_1678865918152.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_radiology_pipeline", "en", "clinical/models")

text = '''Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_radiology_pipeline", "en", "clinical/models")

val text = "Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk                                |   begin |   end | ner_label                 |   confidence |
|---:|:-----------------------------------------|--------:|------:|:--------------------------|-------------:|
|  0 | Bilateral breast                         |       0 |    15 | BodyPart                  |     0.945    |
|  1 | ultrasound                               |      17 |    26 | ImagingTest               |     0.6734   |
|  2 | ovoid mass                               |      78 |    87 | ImagingFindings           |     0.6095   |
|  3 | 0.5 x 0.5 x 0.4                          |     113 |   127 | Measurements              |     0.98158  |
|  4 | cm                                       |     129 |   130 | Units                     |     0.9696   |
|  5 | anteromedial aspect of the left shoulder |     163 |   202 | BodyPart                  |     0.750517 |
|  6 | mass                                     |     210 |   213 | ImagingFindings           |     0.9711   |
|  7 | isoechoic echotexture                    |     228 |   248 | ImagingFindings           |     0.80105  |
|  8 | muscle                                   |     266 |   271 | BodyPart                  |     0.7963   |
|  9 | internal color flow                      |     294 |   312 | ImagingFindings           |     0.477233 |
| 10 | benign fibrous tissue                    |     334 |   354 | ImagingFindings           |     0.524067 |
| 11 | lipoma                                   |     361 |   366 | Disease_Syndrome_Disorder |     0.6081   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_radiology_pipeline|
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