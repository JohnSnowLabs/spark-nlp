---
layout: model
title: Extract Health and Behaviours Problems Entities from Social Determinants of Health Texts
author: John Snow Labs
name: ner_sdoh_health_behaviours_problems_wip
date: 2023-02-24
tags: [clinical, licensed, en, social_determinants, ner, public_health, sdoh, health, behaviours, problems, health_behaviours_problems]
task: Named Entity Recognition
language: en
nav_key: models
edition: Healthcare NLP 4.3.1
spark_version: 3.0
supported: true
annotator: MedicalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model extracts health and behaviours problems related to Social Determinants of Health from various kinds of biomedical documents.

## Predicted Entities

`Diet`, `Mental_Health`, `Obesity`, `Eating_Disorder`, `Sexual_Activity`, `Disability`, `Quality_Of_Life`, `Other_Disease`, `Exercise`, `Communicable_Disease`, `Hyperlipidemia`, `Hypertension`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_health_behaviours_problems_wip_en_4.3.1_3.0_1677198610586.zip){:.button.button-orange}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/ner_sdoh_health_behaviours_problems_wip_en_4.3.1_3.0_1677198610586.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")

tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_sdoh_health_behaviours_problems_wip", "en", "clinical/models")\
    .setInputCols(["sentence", "token","embeddings"])\
    .setOutputCol("ner")

ner_converter = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler, 
    sentence_detector,
    tokenizer,
    clinical_embeddings,
    ner_model,
    ner_converter   
    ])

sample_texts = ["She has not been getting regular exercise and not followed diet for approximately two years due to chronic sciatic pain.",
             "Medical History: The patient is a 32-year-old female who presents with a history of anxiety, depression, bulimia nervosa, elevated cholesterol, and substance abuse.",
               "Pt was intubated atthe scene & currently sedated due to high BP. Also, he is currently on social security disability."]



data = spark.createDataFrame(sample_texts, StringType()).toDF("text")

result = pipeline.fit(data).transform(data)
```
```scala
val document_assembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentence_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "en")
    .setInputCols("document")
    .setOutputCol("sentence")

val tokenizer = new Tokenizer()
    .setInputCols("sentence")
    .setOutputCol("token")

val clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentence", "token"))
    .setOutputCol("embeddings")

val ner_model = MedicalNerModel.pretrained("ner_sdoh_health_behaviours_problems_wip", "en", "clinical/models")
    .setInputCols(Array("sentence", "token","embeddings"))
    .setOutputCol("ner")

val ner_converter = new NerConverterInternal()
    .setInputCols(Array("sentence", "token", "ner"))
    .setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(
    document_assembler, 
    sentence_detector,
    tokenizer,
    clinical_embeddings,
    ner_model,
    ner_converter   
))

val data = Seq("She has not been getting regular exercise for approximately two years due to chronic sciatic pain.").toDS.toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

## Results

```bash
+--------------------+-----+---+---------------+
|chunk               |begin|end|ner_label      |
+--------------------+-----+---+---------------+
|regular exercise    |25   |40 |Exercise       |
|diet                |59   |62 |Diet           |
|chronic sciatic pain|99   |118|Other_Disease  |
|anxiety             |84   |90 |Mental_Health  |
|depression          |93   |102|Mental_Health  |
|bulimia nervosa     |105  |119|Eating_Disorder|
|elevated cholesterol|122  |141|Hyperlipidemia |
|high BP             |56   |62 |Hypertension   |
|disability          |106  |115|Disability     |
+--------------------+-----+---+---------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|ner_sdoh_health_behaviours_problems_wip|
|Compatibility:|Healthcare NLP 4.3.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|3.0 MB|

## Benchmarking

```bash
               label	   tp	   fp	   fn	 total	precision	  recall	      f1
     Quality_Of_Life	127.0	 19.0	  3.0	 130.0	 0.869863	0.976923	0.920290
     Eating_Disorder	 56.0	  5.0	  0.0	  56.0	 0.918033	1.000000	0.957265
             Obesity	 16.0	  2.0	  7.0	  23.0	 0.888889	0.695652	0.780488
            Exercise	103.0	  6.0	  5.0	 108.0	 0.944954	0.953704	0.949309
Communicable_Disease	 61.0	 11.0	  5.0	  66.0	 0.847222	0.924242	0.884058
        Hypertension	 52.0	  0.0	  2.0	  54.0	 1.000000	0.962963	0.981132
       Other_Disease 1068.0	 85.0	 79.0 1147.0	 0.926279	0.931125	0.928696
                Diet	 66.0	 12.0	 15.0	  81.0	 0.846154	0.814815	0.830189
          Disability	 95.0	  1.0	  6.0	 101.0	 0.989583	0.940594	0.964467
       Mental_Health 1020.0	 45.0	134.0	1154.0	 0.957746	0.883882	0.919333
      Hyperlipidemia	 19.0	  1.0	  2.0	  21.0	 0.950000	0.904762	0.926829
     Sexual_Activity	 82.0	 15.0	  6.0	  88.0	 0.845361	0.931818	0.886486
```
