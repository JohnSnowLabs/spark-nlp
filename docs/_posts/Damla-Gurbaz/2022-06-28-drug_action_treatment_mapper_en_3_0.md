---
layout: model
title: Mapping Drugs With Their Corresponding Actions And Treatments
author: John Snow Labs
name: drug_action_treatment_mapper
date: 2022-06-28
tags: [drug, action, treatment, chunk_mapper, clinical, licensed, en]
task: Chunk Mapping
language: en
edition: Healthcare NLP 3.5.3
spark_version: 3.0
supported: true
annotator: ChunkMapperModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This pretrained model maps drugs with their corresponding action and treatment. action refers to the function of the drug in various body systems, treatment refers to which disease the drug is used to treat.

## Predicted Entities

`action`, `treatment`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/drug_action_treatment_mapper_en_3.5.3_3.0_1656398556500.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/drug_action_treatment_mapper_en_3.5.3_3.0_1656398556500.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
.setInputCol('text')\
.setOutputCol('document')

sentence_detector = SentenceDetector()\
.setInputCols(["document"])\
.setOutputCol("sentence")

tokenizer = Tokenizer()\
.setInputCols("sentence")\
.setOutputCol("token")

word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
.setInputCols(["sentence", "token"])\
.setOutputCol("embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_posology_small", "en", "clinical/models")\
.setInputCols(["sentence","token","embeddings"])\
.setOutputCol("ner")\
.setLabelCasing("upper")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")\
.setWhiteList(["DRUG"])

chunkerMapper_action = ChunkMapperModel.pretrained("drug_action_treatment_mapper", "en", "clinical/models")\
.setInputCols(["ner_chunk"])\
.setOutputCol("action_mappings")\
.setRels(["action"])\
.setLowerCase(True)

chunkerMapper_treatment = ChunkMapperModel.pretrained("drug_action_treatment_mapper", "en", "clinical/models")\
.setInputCols(["ner_chunk"])\
.setOutputCol("treatment_mappings")\
.setRels(["treatment"])\
.setLowerCase(True)


pipeline = Pipeline().setStages([document_assembler,
sentence_detector,
tokenizer, 
word_embeddings,
clinical_ner, 
ner_converter,
chunkerMapper_action,
chunkerMapper_treatment])


model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

text = """The patient is a 71-year-old female patient of Dr. X. and she was given Aklis, Dermovate, Aacidexam and Paracetamol."""

pipeline = LightPipeline(model)

result = pipeline.fullAnnotate(text)
```
```scala
val document_assembler = new DocumentAssembler()
.setInputCol("text")
.setOutputCol("document")

val sentence_detector = new SentenceDetector()
.setInputCols("document")
.setOutputCol("sentence")

val tokenizer = new Tokenizer()
.setInputCols("sentence")
.setOutputCol("token")

val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val clinical_ner = MedicalNerModel.pretrained("ner_posology_small","en","clinical/models")
.setInputCols(Array("sentence","token","embeddings"))
.setOutputCol("ner")
.setLabelCasing("upper")

val ner_converter = new NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")
.setWhiteList(Array("DRUG"))

val chunkerMapper_action = ChunkMapperModel.pretrained("drug_action_treatment_mapper", "en", "clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("action_mappings")
.setRels(Array("action"))
.setLowerCase(True)

val chunkerMapper_treatment = ChunkMapperModel.pretrained("drug_action_treatment_mapper", "en", "clinical/models")
.setInputCols(Array("ner_chunk"))
.setOutputCol("treatment_mappings")
.setRels(Array("treatment"))
.setLowerCase(True)


val pipeline = new Pipeline().setStages(Array(
document_assembler,
sentence_detector,
tokenizer, 
word_embeddings,
clinical_ner, 
ner_converter,
chunkerMapper_action,
chunkerMapper_treatment))




val data = Seq("""The patient is a 71-year-old female patient of Dr. X. and she was given Aklis, Dermovate, Aacidexam and Paracetamol.""").toDS.toDF("text")

val result= pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.map_entity.drug_to_action_treatment").predict("""The patient is a 71-year-old female patient of Dr. X. and she was given Aklis, Dermovate, Aacidexam and Paracetamol.""")
```

</div>

## Results

```bash
+-----------+--------------------+-------------------+------------------------------------------------------------+-----------------------------------------------------------------------------+
|Drug       |action_mappings     |treatment_mappings |action_meta                                                 |treatment_meta                                                               |
+-----------+--------------------+-------------------+------------------------------------------------------------+-----------------------------------------------------------------------------+
|Aklis      |cardioprotective    |hyperlipidemia     |hypotensive:::natriuretic                                   |hypertension:::diabetic kidney disease:::cerebrovascular accident:::smoking  |
|Dermovate  |anti-inflammatory   |lupus              |corticosteroids::: dermatological preparations:::very strong|discoid lupus erythematosus:::empeines:::psoriasis:::eczema                  |
|Aacidexam  |antiallergic        |abscess            |antiexudative:::anti-inflammatory:::anti-shock              |brain abscess:::agranulocytosis:::adrenogenital syndrome                     |
|Paracetamol|analgesic           |arthralgia         |anti-inflammatory:::antipyretic:::pain reliever             |period pain:::pain:::sore throat:::headache:::influenza:::toothache          |   
+-----------+--------------------+-------------------+------------------------------------------------------------+-----------------------------------------------------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|drug_action_treatment_mapper|
|Compatibility:|Healthcare NLP 3.5.3+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[ner_chunk]|
|Output Labels:|[mappings]|
|Language:|en|
|Size:|8.4 MB|
