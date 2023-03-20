---
layout: model
title: Pipeline to Detect Radiology Concepts - WIP (biobert)
author: John Snow Labs
name: jsl_rd_ner_wip_greedy_biobert_pipeline
date: 2023-03-20
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

This pretrained pipeline is built on the top of [jsl_rd_ner_wip_greedy_biobert](https://nlp.johnsnowlabs.com/2021/07/26/jsl_rd_ner_wip_greedy_biobert_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_rd_ner_wip_greedy_biobert_pipeline_en_4.3.0_3.2_1679310411354.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/jsl_rd_ner_wip_greedy_biobert_pipeline_en_4.3.0_3.2_1679310411354.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("jsl_rd_ner_wip_greedy_biobert_pipeline", "en", "clinical/models")

text = '''Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("jsl_rd_ner_wip_greedy_biobert_pipeline", "en", "clinical/models")

val text = "Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk             |   begin |   end | ner_label                 |   confidence |
|---:|:----------------------|--------:|------:|:--------------------------|-------------:|
|  0 | Bilateral             |       0 |     8 | Direction                 |     0.9875   |
|  1 | breast                |      10 |    15 | BodyPart                  |     0.6109   |
|  2 | ultrasound            |      17 |    26 | ImagingTest               |     0.7902   |
|  3 | ovoid mass            |      78 |    87 | ImagingFindings           |     0.42185  |
|  4 | 0.5 x 0.5 x 0.4       |     113 |   127 | Measurements              |     0.9406   |
|  5 | cm                    |     129 |   130 | Units                     |     1        |
|  6 | left                  |     190 |   193 | Direction                 |     0.5566   |
|  7 | shoulder              |     195 |   202 | BodyPart                  |     0.6228   |
|  8 | mass                  |     210 |   213 | ImagingFindings           |     0.9463   |
|  9 | isoechoic echotexture |     228 |   248 | ImagingFindings           |     0.4332   |
| 10 | muscle                |     266 |   271 | BodyPart                  |     0.7148   |
| 11 | internal color flow   |     294 |   312 | ImagingFindings           |     0.3726   |
| 12 | benign fibrous tissue |     334 |   354 | ImagingFindings           |     0.484533 |
| 13 | lipoma                |     361 |   366 | Disease_Syndrome_Disorder |     0.8955   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|jsl_rd_ner_wip_greedy_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 4.3.0+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|422.8 MB|

## Included Models

- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverterInternalModel