---
layout: model
title: Detect Living Species
author: John Snow Labs
name: ner_living_species
date: 2022-06-22
tags: [en, ner, clinical, licensed]
task: Named Entity Recognition
language: en
edition: Spark NLP for Healthcare 3.5.3
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Extract living species from clinical texts which is critical to scientific disciplines like medicine, biology, ecology/biodiversity, nutrition and agriculture.

It is trained on the [LivingNER](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/) corpus that is composed of clinical case reports extracted from miscellaneous medical specialties including COVID, oncology, infectious diseases, tropical medicine, urology, pediatrics, and others.

**NOTE :**
1.	The text files were translated from Spanish with a neural machine translation system.
2.	The annotations were translated with the same neural machine translation system.
3.	The translated annotations were transferred to the translated text files using an annotation transfer technology.

## Predicted Entities

`HUMAN`, `SPECIES`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_living_species_en_3.5.3_3.0_1655888659088.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")\
    .setInputCols("sentence","token")\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_living_species", "en","clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")\

ner_converter = NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter   
    ])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame([["""Patient aged 61 years; no known drug allergies, smoker of 63 packs/year, significant active alcoholism, recently diagnosed hypertension. He came to the emergency department approximately 4 days ago with a frontal headache coinciding with a diagnosis of hypertension, for which he was started on antihypertensive treatment. The family reported that they found him "slower" accompanied by behavioural alterations; with no other accompanying symptoms.Physical examination: Glasgow Glasgow 15; neurological examination without focality except for bradypsychia and disorientation in time, person and space. Afebrile. BP: 159/92; heart rate 70 and O2 Sat: 93%; abdominal examination revealed hepatomegaly of two finger widths with no other noteworthy findings. CBC: Legionella antigen and pneumococcus in urine negative."""]]).toDF("text")

result = model.transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare", "en", "clinical/models")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical","en","clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_living_species", "en","clinical/models")
    .setInputCols(Array("sentence", "token", "embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverter()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new PipelineModel().setStages(Array(document_assembler, 
                                                  sentence_detector,
                                                  tokenizer,
                                                  embeddings,
                                                  ner_model,
                                                  ner_converter))

val data = Seq("""Patient aged 61 years; no known drug allergies, smoker of 63 packs/year, significant active alcoholism, recently diagnosed hypertension. He came to the emergency department approximately 4 days ago with a frontal headache coinciding with a diagnosis of hypertension, for which he was started on antihypertensive treatment. The family reported that they found him "slower" accompanied by behavioural alterations; with no other accompanying symptoms.Physical examination: Glasgow Glasgow 15; neurological examination without focality except for bradypsychia and disorientation in time, person and space. Afebrile. BP: 159/92; heart rate 70 and O2 Sat: 93%; abdominal examination revealed hepatomegaly of two finger widths with no other noteworthy findings. CBC: Legionella antigen and pneumococcus in urine negative.""").toDS.toDF("text")

result = model.fit(data).transform(data)
```
</div>

## Results

```bash
+------------+-------+
|ner_chunk   |label  |
+------------+-------+
|Patient     |HUMAN  |
|family      |HUMAN  |
|person      |HUMAN  |
|Legionella  |SPECIES|
|pneumococcus|SPECIES|
+------------+-------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_living_species|
|Compatibility:|Spark NLP for Healthcare 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|15.1 MB|

## References

[https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/](https://temu.bsc.es/livingner/2022/05/03/multilingual-corpus/)

## Benchmarking

```bash
 label         precision  recall  f1-score  support 
 B-HUMAN       0.84       0.96    0.90      2950    
 B-SPECIES     0.73       0.92    0.81      3129    
 I-HUMAN       0.69       0.68    0.69      145     
 I-SPECIES     0.66       0.89    0.76      1166    
 micro-avg     0.76       0.93    0.83      7390    
 macro-avg     0.73       0.86    0.79      7390    
 weighted-avg  0.76       0.93    0.83      7390 
```