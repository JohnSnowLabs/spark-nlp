---
layout: model
title: Relation extraction between dates and clinical entities
author: John Snow Labs
name: re_date_clinical
date: 2021-01-18
task: Relation Extraction
language: en
edition: Spark NLP for Healthcare 2.7.1
spark_version: 2.4
tags: [en, relation_extraction, clinical, licensed]
supported: true
annotator: RelationExtractionModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Relation extraction between date and related other entities. `1` : Shows there is a relation between the date entity and other clinical entities, `0` : Shows there is no relation between the date entity and other clinical entities.

## Predicted Entities

`0`, `1`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_date_clinical_en_2.7.1_2.4_1611000334654.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/re_date_clinical_en_2.7.1_2.4_1611000334654.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

In the table below, `re_date_clinical` RE model, its labels, optimal NER model, and meaningful relation pairs are illustrated.

|     RE MODEL     | RE MODEL LABES | NER MODEL | RE PAIRS |
|:----------------:|:--------------:|:---------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| re_date_clinical |       0,1      |  ner_jsl  | [“date-admission_discharge”, <br>“admission_discharge-date”,<br>“date-alcohol”, <br>“alcohol-date”,<br>“date-allergen”, <br>“allergen-date”,<br>“date-bmi”, <br>“bmi-date”,<br>“date-birth_entity”, <br>“birth_entity-date”,<br>“date-blood_pressure”, <br>“blood_pressure-date”,<br>“date-cerebrovascular_disease”, <br>“cerebrovascular_disease-date”,<br>“date-clinical_dept”, <br>“clinical_dept-date”,<br>“date-communicable_disease”, <br>“communicable_disease-date”,<br>“date-death_entity”, <br>“death_entity-date”,<br>“date-diabetes”, <br>“diabetes-date”,<br>“date-diet”, <br>“diet-date”,<br>“date-disease_syndrome_disorder”, <br>“disease_syndrome_disorder-date”,<br>“date-drug_brandname”, <br>“drug_brandname-date”,<br>“date-drug_ingredient”, <br>“drug_ingredient-date”,<br>“date-ekg_findings”, <br>“ekg_findings-date”,<br>“date-external_body_part_or_region”, <br>“external_body_part_or_region-date”,<br>“date-fetus_newborn”, <br>“fetus_newborn-date”,<br>“date-hdl”, <br>“hdl-date”,<br>“date-heart_disease”, <br>“heart_disease-date”,<br>“date-height”, <br>“height-date”,<br>“date-hyperlipidemia”, <br>“hyperlipidemia-date”,<br>“date-hypertension”, <br>“hypertension-date”,<br>“date-imagingfindings”, <br>“imagingfindings-date”,<br>“date-imaging_technique”, <br>“imaging_technique-date”,<br>“date-injury_or_poisoning”, <br>“injury_or_poisoning-date”,<br>“date-internal_organ_or_component”, <br>“internal_organ_or_component-date”,<br>“date-kidney_disease”, <br>“kidney_disease-date”,<br>“date-ldl”, <br>“ldl-date”,<br>“date-modifier”, <br>“modifier-date”,<br>“date-o2_saturation”, <br>“o2_saturation-date”,<br>“date-obesity”, <br>“obesity-date”,<br>“date-oncological”, <br>“oncological-date”,<br>“date-overweight”, <br>“overweight-date”,<br>“date-oxygen_therapy”, <br>“oxygen_therapy-date”,<br>“date-pregnancy”, <br>“pregnancy-date”,<br>“date-procedure”, <br>“procedure-date”,<br>“date-psychological_condition”, <br>“psychological_condition-date”,<br>“date-pulse”, <br>“pulse-date”,<br>“date-respiration”, <br>“respiration-date”,<br>“date-smoking”, <br>“smoking-date”,<br>“date-substance”, <br>“substance-date”,<br>“date-substance_quantity”, <br>“substance_quantity-date”,<br>“date-symptom”, <br>“symptom-date”,<br>“date-temperature”, <br>“temperature-date”,<br>“date-test”, <br>“test-date”,<br>“date-test_result”, <br>“test_result-date”,<br>“date-total_cholesterol”, <br>“total_cholesterol-date”,<br>“date-treatment”, <br>“treatment-date”,<br>“date-triglycerides”, <br>“triglycerides-date”,<br>“date-vs_finding”, <br>“vs_finding-date”,<br>“date-vaccine”, <br>“vaccine-date”,<br>“date-vital_signs_header”, <br>“vital_signs_header-date”,<br>“date-weight”, <br>“weight-date”,<br>“time-admission_discharge”, <br>“admission_discharge-time”,<br>“time-alcohol”, <br>“alcohol-time”,<br>“time-allergen”, <br>“allergen-time”,<br>“time-bmi”, <br>“bmi-time”,<br>“time-birth_entity”, <br>“birth_entity-time”,<br>“time-blood_pressure”,<br>“blood_pressure-time”,<br>“time-cerebrovascular_disease”, <br>“cerebrovascular_disease-time”,<br>“time-clinical_dept”, <br>“clinical_dept-time”,<br>“time-communicable_disease”, <br>“communicable_disease-time”,<br>“time-death_entity”, <br>“death_entity-time”,<br>“time-diabetes”, <br>“diabetes-time”,<br>“time-diet”, <br>“diet-time”,<br>“time-disease_syndrome_disorder”, <br>“disease_syndrome_disorder-time”,<br>“time-drug_brandname”, <br>“drug_brandname-time”,<br>“time-drug_ingredient”, <br>“drug_ingredient-time”,<br>“time-ekg_findings”, <br>“ekg_findings-time”,<br>“time-external_body_part_or_region”, <br>“external_body_part_or_region-time”,<br>“time-fetus_newborn”, <br>“fetus_newborn-time”,<br>“time-hdl”, <br>“hdl-time”,<br>“time-heart_disease”, <br>“heart_disease-time”,<br>“time-height”, <br>“height-time”,<br>“time-hyperlipidemia”, <br>“hyperlipidemia-time”,<br>“time-hypertension”, <br>“hypertension-time”,<br>“time-imagingfindings”, <br>“imagingfindings-time”,<br>“time-imaging_technique”, <br>“imaging_technique-time”,<br>“time-injury_or_poisoning”, <br>“injury_or_poisoning-time”,<br>“time-internal_organ_or_component”, <br>“internal_organ_or_component-time”,<br>“time-kidney_disease”, <br>“kidney_disease-time”,<br>“time-ldl”, <br>“ldl-time”,<br>“time-modifier”, <br>“modifier-time”,<br>“time-o2_saturation”, <br>“o2_saturation-time”,<br>“time-obesity”,<br>“obesity-time”,<br>“time-oncological”, <br>“oncological-time”,<br>“time-overweight”, <br>“overweight-time”,<br>“time-oxygen_therapy”, <br>“oxygen_therapy-time”,<br>“time-pregnancy”, <br>“pregnancy-time”,<br>“time-procedure”, <br>“procedure-time”,<br>“time-psychological_condition”, <br>“psychological_condition-time”,<br>“time-pulse”, <br>“pulse-time”,<br>“time-respiration”, <br>“respiration-time”,<br>“time-smoking”, <br>“smoking-time”,<br>“time-substance”, <br>“substance-time”,<br>“time-substance_quantity”, <br>“substance_quantity-time”,<br>“time-symptom”, <br>“symptom-time”,<br>“time-temperature”, <br>“temperature-time”,<br>“time-test”, <br>“test-time”,<br>“time-test_result”, <br>“test_result-time”,<br>“time-total_cholesterol”, <br>“total_cholesterol-time”,<br>“time-treatment”, <br>“treatment-time”,<br>“time-triglycerides”, <br>“triglycerides-time”,<br>“time-vs_finding”, <br>“vs_finding-time”,<br>“time-vaccine”, <br>“vaccine-time”,<br>“time-vital_signs_header”, <br>“vital_signs_header-time”,<br>“time-weight”, <br>“weight-time”,<br>“relativedate-admission_discharge”, <br>“admission_discharge-relativedate”,<br>“relativedate-alcohol”, <br>“alcohol-relativedate”,<br>“relativedate-allergen”, <br>“allergen-relativedate”,<br>“relativedate-bmi”, <br>“bmi-relativedate”,<br>“relativedate-birth_entity”, <br>“birth_entity-relativedate”,<br>“relativedate-blood_pressure”, <br>“blood_pressure-relativedate”,<br>“relativedate-cerebrovascular_disease”, <br>“cerebrovascular_disease-relativedate”,<br>“relativedate-clinical_dept”, <br>“clinical_dept-relativedate”,<br>“relativedate-communicable_disease”, <br>“communicable_disease-relativedate”,<br>“relativedate-death_entity”, <br>“death_entity-relativedate”,<br>“relativedate-diabetes”, <br>“diabetes-relativedate”,<br>“relativedate-diet”, <br>“diet-relativedate”,<br>“relativedate-disease_syndrome_disorder”, <br>“disease_syndrome_disorder-relativedate”,<br>“relativedate-drug_brandname”, <br>“drug_brandname-relativedate”,<br>“relativedate-drug_ingredient”, <br>“drug_ingredient-relativedate”,<br>“relativedate-ekg_findings”, <br>“ekg_findings-relativedate”,<br>“relativedate-external_body_part_or_region”, <br>“external_body_part_or_region-relativedate”,<br>“relativedate-fetus_newborn”, <br>“fetus_newborn-relativedate”,<br>“relativedate-hdl”, <br>“hdl-relativedate”,<br>“relativedate-heart_disease”, <br>“heart_disease-relativedate”,<br>“relativedate-height”, <br>“height-relativedate”,<br>“relativedate-hyperlipidemia”, <br>“hyperlipidemia-relativedate”,<br>“relativedate-hypertension”, <br>“hypertension-relativedate”,<br>“relativedate-imagingfindings”, <br>“imagingfindings-relativedate”,<br>“relativedate-imaging_technique”, <br>“imaging_technique-relativedate”,<br>“relativedate-injury_or_poisoning”, <br>“injury_or_poisoning-relativedate”,<br>“relativedate-internal_organ_or_component”, <br>“internal_organ_or_component-relativedate”,<br>“relativedate-kidney_disease”, <br>“kidney_disease-relativedate”,<br>“relativedate-ldl”, <br>“ldl-relativedate”,<br>“relativedate-modifier”, <br>“modifier-relativedate”,<br>“relativedate-o2_saturation”, <br>“o2_saturation-relativedate”,<br>“relativedate-obesity”, <br>“obesity-relativedate”,<br>“relativedate-oncological”, <br>“oncological-relativedate”,<br>“relativedate-overweight”, <br>“overweight-relativedate”,<br>“relativedate-oxygen_therapy”, <br>“oxygen_therapy-relativedate”,<br>“relativedate-pregnancy”, <br>“pregnancy-relativedate”,<br>“relativedate-procedure”, <br>“procedure-relativedate”,<br>“relativedate-psychological_condition”, <br>“psychological_condition-relativedate”,<br>“relativedate-pulse”, <br>“pulse-relativedate”,<br>“relativedate-respiration”, <br>“respiration-relativedate”,<br>“relativedate-smoking”, <br>“smoking-relativedate”,<br>“relativedate-substance”, <br>“substance-relativedate”,<br>“relativedate-substance_quantity”, <br>“substance_quantity-relativedate”,<br>“relativedate-symptom”, <br>“symptom-relativedate”,<br>“relativedate-temperature”, <br>“temperature-relativedate”,<br>“relativedate-test”, <br>“test-relativedate”,<br>“relativedate-test_result”, <br>“test_result-relativedate”,<br>“relativedate-total_cholesterol”, <br>“total_cholesterol-relativedate”,<br>“relativedate-treatment”, <br>“treatment-relativedate”,<br>“relativedate-triglycerides”, <br>“triglycerides-relativedate”,<br>“relativedate-vs_finding”, <br>“vs_finding-relativedate”,<br>“relativedate-vaccine”, <br>“vaccine-relativedate”,<br>“relativedate-vital_signs_header”, <br>“vital_signs_header-relativedate”,<br>“relativedate-weight”,<br>“weight-relativedate”,<br>“relativetime-admission_discharge”, <br>“admission_discharge-relativetime”,<br>“relativetime-alcohol”, <br>“alcohol-relativetime”,<br>“relativetime-allergen”, <br>“allergen-relativetime”,<br>“relativetime-bmi”, <br>“bmi-relativetime”,<br>“relativetime-birth_entity”, <br>“birth_entity-relativetime”,<br>“relativetime-blood_pressure”, <br>“blood_pressure-relativetime”,<br>“relativetime-cerebrovascular_disease”, <br>“cerebrovascular_disease-relativetime”,<br>“relativetime-clinical_dept”, <br>“clinical_dept-relativetime”,<br>“relativetime-communicable_disease”, <br>“communicable_disease-relativetime”,<br>“relativetime-death_entity”, <br>“death_entity-relativetime”,<br>“relativetime-diabetes”, <br>“diabetes-relativetime”,<br>“relativetime-diet”, <br>“diet-relativetime”,<br>“relativetime-disease_syndrome_disorder”, <br>“disease_syndrome_disorder-relativetime”,<br>“relativetime-drug_brandname”, <br>“drug_brandname-relativetime”,<br>“relativetime-drug_ingredient”, <br>“drug_ingredient-relativetime”,<br>“relativetime-ekg_findings”, <br>“ekg_findings-relativetime”,<br>“relativetime-external_body_part_or_region”, <br>“external_body_part_or_region-relativetime”,<br>“relativetime-fetus_newborn”, <br>“fetus_newborn-relativetime”,<br>“relativetime-hdl”, <br>“hdl-relativetime”,<br>“relativetime-heart_disease”, <br>“heart_disease-relativetime”,<br>“relativetime-height”, <br>“height-relativetime”,<br>“relativetime-hyperlipidemia”, <br>“hyperlipidemia-relativetime”,<br>“relativetime-hypertension”, <br>“hypertension-relativetime”,<br>“relativetime-imagingfindings”, <br>“imagingfindings-relativetime”,<br>“relativetime-imaging_technique”, <br>“imaging_technique-relativetime”,<br>“relativetime-injury_or_poisoning”, <br>“injury_or_poisoning-relativetime”,<br>“relativetime-internal_organ_or_component”, <br>“internal_organ_or_component-relativetime”,<br>“relativetime-kidney_disease”, <br>“kidney_disease-relativetime”,<br>“relativetime-ldl”, <br>“ldl-relativetime”,<br>“relativetime-modifier”, <br>“modifier-relativetime”,<br>“relativetime-o2_saturation”, <br>“o2_saturation-relativetime”,<br>“relativetime-obesity”, <br>“obesity-relativetime”,<br>“relativetime-oncological”, <br>“oncological-relativetime”,<br>“relativetime-overweight”, <br>“overweight-relativetime”,<br>“relativetime-oxygen_therapy”, <br>“oxygen_therapy-relativetime”,<br>“relativetime-pregnancy”, <br>“pregnancy-relativetime”,<br>“relativetime-procedure”, <br>“procedure-relativetime”,<br>“relativetime-psychological_condition”, <br>“psychological_condition-relativetime”,<br>“relativetime-pulse”, <br>“pulse-relativetime”,<br>“relativetime-respiration”, <br>“respiration-relativetime”,<br>“relativetime-smoking”, <br>“smoking-relativetime”,<br>“relativetime-substance”, <br>“substance-relativetime”,<br>“relativetime-substance_quantity”, <br>“substance_quantity-relativetime”,<br>“relativetime-symptom”, <br>“symptom-relativetime”,<br>“relativetime-temperature”, <br>“temperature-relativetime”,<br>“relativetime-test”, <br>“test-relativetime”,<br>“relativetime-test_result”, <br>“test_result-relativetime”,<br>“relativetime-total_cholesterol”, <br>“total_cholesterol-relativetime”,<br>“relativetime-treatment”, <br>“treatment-relativetime”,<br>“relativetime-triglycerides”, <br>“triglycerides-relativetime”,<br>“relativetime-vs_finding”, <br>“vs_finding-relativetime”,<br>“relativetime-vaccine”, <br>“vaccine-relativetime”,<br>“relativetime-vital_signs_header”, <br>“vital_signs_header-relativetime”,<br>“relativetime-weight”,<br>“weight-relativetime”] |


Use as part of an nlp pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, PerceptronModel, DependencyParserModel, WordEmbeddingsModel, NerDLModel, NerConverter, RelationExtractionModel.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sentencer = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentences")

tokenizer = Tokenizer()\
    .setInputCols(["sentences"])\
    .setOutputCol("tokens")
  
word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("embeddings")

pos_tagger = PerceptronModel()\
    .pretrained("pos_clinical", "en", "clinical/models") \
    .setInputCols(["sentences", "tokens"])\
    .setOutputCol("pos_tags")

ner_tagger = MedicalNerModel().pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags")

ner_chunker = NerConverterInternal()\
    .setInputCols(["sentences", "tokens", "ner_tags"])\
    .setOutputCol("ner_chunks")

dependency_parser = DependencyParserModel()\
    .pretrained("dependency_conllu", "en")\
    .setInputCols(["sentences", "pos_tags", "tokens"])\
    .setOutputCol("dependencies")

re_model = RelationExtractionModel().pretrained("re_date_clinical", "en", "clinical/models")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(3)\
    .setPredictionThreshold(0.9)\
    .setRelationPairs(["test-date", "symptom-date"]) # Possible relation pairs. Default: All Relations.

nlp_pipeline = Pipeline(stages=[documenter, sentencer,tokenizer, word_embeddings, pos_tagger, ner_tagger, ner_chunker, dependency_parser, re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate('''This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.''')
```

```scala
val documenter = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val sentencer = new SentenceDetector()
    .setInputCols(["document"])
    .setOutputCol("sentences")

val tokenizer = new Tokenizer()
    .setInputCols("sentences")
    .setOutputCol("tokens")
  
val word_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("embeddings")

val pos_tagger = PerceptronModel()
    .pretrained("pos_clinical", "en", "clinical/models")
    .setInputCols(Array("sentences", "tokens"))
    .setOutputCol("pos_tags")

val ner_tagger = MedicalNerModel().pretrained("jsl_ner_wip_greedy_clinical","en","clinical/models")
    .setInputCols(Array("sentences", "tokens", "embeddings"))
    .setOutputCol("ner_tags")

val ner_chunker = new NerConverterInternal()
    .setInputCols(Array("sentences", "tokens", "ner_tags"))
    .setOutputCol("ner_chunks")

val dependency_parser = DependencyParserModel()
    .pretrained("dependency_conllu", "en")
    .setInputCols(Array("sentences", "pos_tags", "tokens"))
    .setOutputCol("dependencies")

val re_model = RelationExtractionModel()
    .pretrained("re_date", "en", "clinical/models")
    .setInputCols(Array("embeddings", "pos_tags", "ner_chunks", "dependencies"))
    .setOutputCol("relations")
    .setMaxSyntacticDistance(3) #default: 0 
    .setPredictionThreshold(0.9) #default: 0.5 
    .setRelationPairs(Array("test-date", "symptom-date")) # Possible relation pairs. Default: All Relations.

val nlpPipeline = new Pipeline().setStages(Array(documenter, sentencer,tokenizer, word_embeddings, pos_tagger, ner_tagger, ner_chunker, dependency_parser, re_model))

val result = pipeline.fit(Seq.empty[String]).transform(data)

val annotations = light_pipeline.fullAnnotate("""This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.""")
```

</div>

## Results

```bash
|   | relations | entity1 | entity1_begin | entity1_end | chunk1                                   | entity2 | entity2_end | entity2_end | chunk2  | confidence |
|---|-----------|---------|---------------|-------------|------------------------------------------|---------|-------------|-------------|---------|------------|
| 0 | 1         | Test    | 24            | 25          | CT                                       | Date    | 31          | 37          | 1/12/95 | 1.0        |
| 1 | 1         | Symptom | 45            | 84          | progressive memory and cognitive decline | Date    | 92          | 98          | 8/11/94 | 1.0        |
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_date_clinical|
|Type:|re|
|Compatibility:|Spark NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|
|Dependencies:|embeddings_clinical|

## Data Source

Trained on data gathered and manually annotated by John Snow Labs

## Benchmarking

```bash
label recall  precision  f1   
0     0.74    0.71       0.72
1     0.94    0.95       0.94
```