---
layout: model
title: Pipeline to Detect Drug Information
author: John Snow Labs
name: ner_posology_large_biobert_pipeline
date: 2022-03-21
tags: [licensed, ner, clinical, drug, en]
task: [Named Entity Recognition, Pipeline Healthcare]
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


This pretrained pipeline is built on the top of [ner_posology_large_biobert](https://nlp.johnsnowlabs.com/2021/04/01/ner_posology_large_biobert_en.html) model.


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/){:.button.button-orange.button-orange-trans.arr.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_POSOLOGY.ipynb){:.button.button-orange.button-orange-trans.arr.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_large_biobert_pipeline_en_3.4.1_3.0_1647873520729.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_posology_large_biobert_pipeline_en_3.4.1_3.0_1647873520729.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use






<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}


```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("ner_posology_large_biobert_pipeline", "en", "clinical/models")

pipeline.fullAnnotate('The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2; coronary artery disease; chronic renal insufficiency; peripheral vascular disease, also secondary to diabetes; who was originally admitted to an outside hospital for what appeared to be acute paraplegia, lower extremities. She did receive a course of Bactrim for 14 days for UTI. Evidently, at some point in time, the patient was noted to develop a pressure-type wound on the sole of her left foot and left great toe. She was also noted to have a large sacral wound; this is in a similar location with her previous laminectomy, and this continues to receive daily care. The patient was transferred secondary to inability to participate in full physical and occupational therapy and continue medical management of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., Lantus 40 units subcutaneously at bedtime, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, Norvasc 10 mg daily, Lexapro 20 mg daily, aspirin 81 mg daily, Senna 2 tablets p.o. q.a.m., Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily, and Bactrim DS b.i.d.')
```
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline

val pipeline = new PretrainedPipeline("ner_posology_large_biobert_pipeline", "en", "clinical/models")

pipeline.fullAnnotate("The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2; coronary artery disease; chronic renal insufficiency; peripheral vascular disease, also secondary to diabetes; who was originally admitted to an outside hospital for what appeared to be acute paraplegia, lower extremities. She did receive a course of Bactrim for 14 days for UTI. Evidently, at some point in time, the patient was noted to develop a pressure-type wound on the sole of her left foot and left great toe. She was also noted to have a large sacral wound; this is in a similar location with her previous laminectomy, and this continues to receive daily care. The patient was transferred secondary to inability to participate in full physical and occupational therapy and continue medical management of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., Lantus 40 units subcutaneously at bedtime, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, Norvasc 10 mg daily, Lexapro 20 mg daily, aspirin 81 mg daily, Senna 2 tablets p.o. q.a.m., Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily, and Bactrim DS b.i.d.")
```
</div>


## Results


```bash
+--------------+---------+
|chunks        |entities |
+--------------+---------+
|Bactrim       |DRUG     |
|for 14 days   |DURATION |
|Fragmin       |DRUG     |
|5000 units    |STRENGTH |
|subcutaneously|ROUTE    |
|daily         |FREQUENCY|
|Xenaderm      |DRUG     |
|topically     |ROUTE    |
|b.i.d         |FREQUENCY|
|Lantus        |DRUG     |
|40 units      |STRENGTH |
|subcutaneously|ROUTE    |
|at bedtime    |FREQUENCY|
|OxyContin     |DRUG     |
|30 mg         |STRENGTH |
|p.o           |ROUTE    |
|folic acid    |DRUG     |
|1 mg          |STRENGTH |
|daily         |FREQUENCY|
|levothyroxine |DRUG     |
+--------------+---------+
only showing top 20 rows
```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|ner_posology_large_biobert_pipeline|
|Type:|pipeline|
|Compatibility:|Healthcare NLP 3.4.1+|
|License:|Licensed|
|Edition:|Official|
|Language:|en|
|Size:|421.9 MB|


## Included Models


- DocumentAssembler
- SentenceDetectorDLModel
- TokenizerModel
- BertEmbeddings
- MedicalNerModel
- NerConverter
<!--stackedit_data:
eyJoaXN0b3J5IjpbNDU5ODgyNzk4XX0=
-->