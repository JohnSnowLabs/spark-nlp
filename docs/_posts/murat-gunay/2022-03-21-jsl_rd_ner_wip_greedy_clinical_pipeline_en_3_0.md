---
layout: model
title: Pipeline to Detect Radiology Concepts (WIP Greedy)
author: John Snow Labs
name: jsl_rd_ner_wip_greedy_clinical_pipeline
date: 2022-03-21
tags: [licensed, ner, wip, clinical, radiology, en]
task: Named Entity Recognition
language: en
edition: Healthcare NLP 3.4.1
spark_version: 3.0
supported: true
annotator: PipelineModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained pipeline is built on the top of [jsl_rd_ner_wip_greedy_clinical](https://nlp.johnsnowlabs.com/2021/04/01/jsl_rd_ner_wip_greedy_clinical_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/jsl_rd_ner_wip_greedy_clinical_pipeline_en_3.4.1_3.0_1647867631313.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("jsl_rd_ner_wip_greedy_clinical_pipeline", "en", "clinical/models")

pipeline.annotate("Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma.")
```
```scala
val pipeline = new PretrainedPipeline("jsl_rd_ner_wip_greedy_clinical_pipeline", "en", "clinical/models")

pipeline.annotate("Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma.")
```
</div>

## Results

```bash
+---------------------+-------------------------+
|chunks               |entities                 |
+---------------------+-------------------------+
|Bilateral            |Direction                |
|breast               |BodyPart                 |
|ultrasound           |ImagingTest              |
|ovoid mass           |Symptom                  |
|anteromedial aspect  |Direction                |
|left                 |Direction                |
|shoulder             |BodyPart                 |
|mass                 |Symptom                  |
|isoechoic echotexture|ImagingFindings          |
|adjacent muscle      |BodyPart                 |
|internal color flow  |Symptom                  |
|benign fibrous tissue|Symptom                  |
|lipoma               |Disease_Syndrome_Disorder|
+---------------------+-------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|jsl_rd_ner_wip_greedy_clinical_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
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
- NerConverter
