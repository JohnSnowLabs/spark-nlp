---
layout: model
title: Pipeline to Detect Drug Information (Large)
author: John Snow Labs
name: ner_posology_large_pipeline
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

This pretrained pipeline is built on the top of [ner_posology_large](https://nlp.johnsnowlabs.com/2021/03/31/ner_posology_large_en.html) model.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_large_pipeline_en_4.3.0_3.2_1678869355529.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_posology_large_pipeline_en_4.3.0_3.2_1678869355529.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_posology_large_pipeline", "en", "clinical/models")

text = '''The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2; coronary artery disease; chronic renal insufficiency; peripheral vascular disease, also secondary to diabetes; who was originally admitted to an outside hospital for what appeared to be acute paraplegia, lower extremities. She did receive a course of Bactrim for 14 days for UTI. Evidently, at some point in time, the patient was noted to develop a pressure-type wound on the sole of her left foot and left great toe. She was also noted to have a large sacral wound; this is in a similar location with her previous laminectomy, and this continues to receive daily care. The patient was transferred secondary to inability to participate in full physical and occupational therapy and continue medical management of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., Lantus 40 units subcutaneously at bedtime, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, Norvasc 10 mg daily, Lexapro 20 mg daily, aspirin 81 mg daily, Senna 2 tablets p.o. q.a.m., Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily, and Bactrim DS b.i.d.'''

result = pipeline.fullAnnotate(text)
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_posology_large_pipeline", "en", "clinical/models")

val text = "The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2; coronary artery disease; chronic renal insufficiency; peripheral vascular disease, also secondary to diabetes; who was originally admitted to an outside hospital for what appeared to be acute paraplegia, lower extremities. She did receive a course of Bactrim for 14 days for UTI. Evidently, at some point in time, the patient was noted to develop a pressure-type wound on the sole of her left foot and left great toe. She was also noted to have a large sacral wound; this is in a similar location with her previous laminectomy, and this continues to receive daily care. The patient was transferred secondary to inability to participate in full physical and occupational therapy and continue medical management of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., Lantus 40 units subcutaneously at bedtime, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, Norvasc 10 mg daily, Lexapro 20 mg daily, aspirin 81 mg daily, Senna 2 tablets p.o. q.a.m., Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily, and Bactrim DS b.i.d."

val result = pipeline.fullAnnotate(text)
```
</div>

## Results

```bash
|    | ner_chunk      |   begin |   end | ner_label   |   confidence |
|---:|:---------------|--------:|------:|:------------|-------------:|
|  0 | insulin        |      59 |    65 | DRUG        |     0.9752   |
|  1 | Bactrim        |     346 |   352 | DRUG        |     0.9994   |
|  2 | for 14 days    |     354 |   364 | DURATION    |     0.796067 |
|  3 | Fragmin        |     925 |   931 | DRUG        |     0.9995   |
|  4 | 5000 units     |     933 |   942 | DOSAGE      |     0.6773   |
|  5 | subcutaneously |     944 |   957 | ROUTE       |     0.9987   |
|  6 | daily          |     959 |   963 | FREQUENCY   |     0.999    |
|  7 | Xenaderm       |     966 |   973 | DRUG        |     0.8853   |
|  8 | topically      |     985 |   993 | ROUTE       |     0.9916   |
|  9 | b.i.d          |     995 |   999 | FREQUENCY   |     0.995    |
| 10 | Lantus         |    1003 |  1008 | DRUG        |     0.9994   |
| 11 | 40 units       |    1010 |  1017 | DOSAGE      |     0.86805  |
| 12 | subcutaneously |    1019 |  1032 | ROUTE       |     0.9986   |
| 13 | at bedtime     |    1034 |  1043 | FREQUENCY   |     0.84895  |
| 14 | OxyContin      |    1046 |  1054 | DRUG        |     0.9875   |
| 15 | 30 mg          |    1056 |  1060 | STRENGTH    |     0.97695  |
| 16 | p.o.           |    1062 |  1065 | ROUTE       |     0.8367   |
| 17 | q.12 h         |    1067 |  1072 | FREQUENCY   |     0.93305  |
| 18 | folic acid     |    1076 |  1085 | DRUG        |     0.9569   |
| 19 | 1 mg           |    1087 |  1090 | STRENGTH    |     0.83715  |
| 20 | daily          |    1092 |  1096 | FREQUENCY   |     0.9998   |
| 21 | levothyroxine  |    1099 |  1111 | DRUG        |     0.9794   |
| 22 | 0.1 mg         |    1113 |  1118 | STRENGTH    |     0.9325   |
| 23 | p.o.           |    1120 |  1123 | ROUTE       |     0.6783   |
| 24 | daily          |    1125 |  1129 | FREQUENCY   |     0.9925   |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_posology_large_pipeline|
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