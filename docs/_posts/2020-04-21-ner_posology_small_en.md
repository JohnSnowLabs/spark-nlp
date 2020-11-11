---
layout: model
title: Posology Extractor Small
author: John Snow Labs
name: ner_posology_small
class: NerDLModel
language: en
repository: clinical/models
date: 2020-04-21
tags: [clinical,ner,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---
 
{:.h2_title}
## Description

Pretrained named entity recognition deep learning model for posology, this NER is trained with the *embeddings_clinical* word embeddings model, so be sure to use the same embeddings in the pipeline

## Predicted Entities 

DOSAGE,DRUG,DURATION,FORM,FREQUENCY,ROUTE,STRENGTH

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_POSOLOGY/){:.button.button-orange}{:target="_blank"}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_posology_small_en_2.4.2_2.4_1587513301751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
clinical_ner = NerDLModel.pretrained("ner_posology_small","en","clinical/models")\
	.setInputCols("sentence","token","word_embeddings")\
	.setOutputCol("ner")

nlp_pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    embeddings_clinical,
    clinical_ner,
    ner_converter])

model = LightPipeline(nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text")))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": [
    """The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2; coronary artery disease; chronic renal insufficiency; peripheral vascular disease, also secondary to diabetes; who was originally admitted to an outside hospital for what appeared to be acute paraplegia, lower extremities. She did receive a course of Bactrim for 14 days for UTI. Evidently, at some point in time, the patient was noted to develop a pressure-type wound on the sole of her left foot and left great toe. She was also noted to have a large sacral wound; this is in a similar location with her previous laminectomy, and this continues to receive daily care. The patient was transferred secondary to inability to participate in full physical and occupational therapy and continue medical management of her diabetes, the sacral decubitus, left foot pressure wound, and associated complications of diabetes. She is given Fragmin 5000 units subcutaneously daily, Xenaderm to wounds topically b.i.d., Lantus 40 units subcutaneously at bedtime, OxyContin 30 mg p.o. q.12 h., folic acid 1 mg daily, levothyroxine 0.1 mg p.o. daily, Prevacid 30 mg daily, Avandia 4 mg daily, Norvasc 10 mg daily, Lexapro 20 mg daily, aspirin 81 mg daily, Senna 2 tablets p.o. q.a.m., Neurontin 400 mg p.o. t.i.d., Percocet 5/325 mg 2 tablets q.4 h. p.r.n., magnesium citrate 1 bottle p.o. p.r.n., sliding scale coverage insulin, Wellbutrin 100 mg p.o. daily, and Bactrim DS b.i.d."""
]})))

```

```scala
val model = NerDLModel.pretrained("ner_posology_small","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
```
</div>

## Results
```bash
+--------------+---------+
|chunk         |ner      |
+--------------+---------+
|insulin       |DRUG     |
|Bactrim       |DRUG     |
|for 14 days   |DURATION |
|Fragmin       |DRUG     |
|5000 units    |DOSAGE   |
|subcutaneously|ROUTE    |
|daily         |FREQUENCY|
|Xenaderm      |DRUG     |
|topically     |ROUTE    |
|b.i.d.,       |FREQUENCY|
|Lantus        |DRUG     |
|40 units      |DOSAGE   |
|subcutaneously|ROUTE    |
|at bedtime    |FREQUENCY|
|OxyContin     |DRUG     |
|30 mg         |STRENGTH |
|p.o           |ROUTE    |
|q.12 h        |FREQUENCY|
|folic acid    |DRUG     |
|1 mg          |STRENGTH |
+--------------+---------+
only showing top 20 rows
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| Name:           | ner_posology_small               |
| Type:    | NerDLModel                       |
| Compatibility:  | Spark NLP 2.4.2+                           |
| License:        | Licensed                         |
|Edition:|Official|                       |
|Input labels:         | sentence, token, word_embeddings |
|Output labels:        | ner                              |
| Language:       | en                               |
| Case sensitive: | False                            |
| Dependencies:  | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on the 2018 i2b2 dataset (no FDA) with *embeddings_clinical*.
https://www.i2b2.org/NLP/Medication