---
layout: model
title: Pipeline to Detect PHI for Deidentification (Enriched)
author: John Snow Labs
name: ner_deid_enriched_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, deidentification, enriched, en]
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

This pretrained pipeline is built on the top of [ner_deid_enriched](https://nlp.johnsnowlabs.com/2021/03/31/ner_deid_enriched_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_enriched_pipeline_en_3.4.1_3.0_1647867856711.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_deid_enriched_pipeline_en_3.4.1_3.0_1647867856711.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
pipeline = PretrainedPipeline("ner_deid_enriched_pipeline", "en", "clinical/models")

pipeline.annotate("HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Remedy City Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003. HOSPITAL COURSE: Mr. Smith presented to the Remedy City Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same.")
```
```scala
val pipeline = new PretrainedPipeline("ner_deid_enriched_pipeline", "en", "clinical/models")

pipeline.annotate("HISTORY OF PRESENT ILLNESS: Mr. Smith is a 60-year-old white male veteran with multiple comorbidities, who has a history of bladder cancer diagnosed approximately two years ago by the VA Hospital. He underwent a resection there. He was to be admitted to the Remedy City Hospital for cystectomy. He was seen in Urology Clinic and Radiology Clinic on 02/04/2003. HOSPITAL COURSE: Mr. Smith presented to the Remedy City Hospital in anticipation for Urology surgery. On evaluation, EKG, echocardiogram was abnormal, a Cardiology consult was obtained. A cardiac adenosine stress MRI was then proceeded, same was positive for inducible ischemia, mild-to-moderate inferolateral subendocardial infarction with peri-infarct ischemia. In addition, inducible ischemia seen in the inferior lateral septum. Mr. Smith underwent a left heart catheterization, which revealed two vessel coronary artery disease. The RCA, proximal was 95% stenosed and the distal 80% stenosed. The mid LAD was 85% stenosed and the distal LAD was 85% stenosed. There was four Multi-Link Vision bare metal stents placed to decrease all four lesions to 0%. Following intervention, Mr. Smith was admitted to Cardiology Service under the direction of Dr. Hart. Mr. Smith had a noncomplicated post-intervention hospital course. He was stable for discharge home on 02/07/2003 with instructions to take Plavix daily for one month and Urology is aware of the same.")
```
</div>

## Results

```bash
+--------------------+---------+
|chunk               |ner_label|
+--------------------+---------+
|Smith               |PATIENT  |
|VA Hospital         |HOSPITAL |
|Remedy City Hospital|HOSPITAL |
|02/04/2003          |DATE     |
|Smith               |PATIENT  |
|Remedy City Hospital|HOSPITAL |
|Smith               |PATIENT  |
|Smith               |PATIENT  |
|Hart                |DOCTOR   |
|Smith               |PATIENT  |
|02/07/2003          |DATE     |
+--------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_deid_enriched_pipeline|
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
