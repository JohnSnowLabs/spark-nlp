---
layout: model
title: Extract relations between problem, test, and findings in reports
author: John Snow Labs
name: re_test_problem_finding
date: 2021-04-19
tags: [en, relation_extraction, licensed, clinical]
task: Relation Extraction
language: en
edition: Healthcare NLP 2.7.1
spark_version: 2.4
supported: true
annotator: RelationExtractionModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Find relations between diagnosis, tests and imaging findings in radiology reports. `1` : The two entities are related. `0` : The two entities are not related

## Predicted Entities

`0`, `1`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_RADIOLOGY/){:.button.button-orange}
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_test_problem_finding_en_2.7.1_2.4_1618830922197.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

In the table below, `re_test_problem_finding` RE model, its labels, optimal NER model, and meaningful relation pairs are illustrated.

 |         RE MODEL        | RE MODEL LABES | NER MODEL | RE PAIRS                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
 |:-----------------------:|:--------------:|:---------:|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
 | re_test_problem_finding |       0,1      |  ner_jsl  | [“test-cerebrovascular_disease”, <br>“cerebrovascular_disease-test”,<br>“test-communicable_disease”, <br>“communicable_disease-test”,<br>“test-diabetes”, “diabetes-test”,<br>“test-disease_syndrome_disorder”, <br>“disease_syndrome_disorder-test”,<br>“test-heart_disease”, <br>“heart_disease-test”,<br>“test-hyperlipidemia”, <br>“hyperlipidemia-test”,<br>“test-hypertension”, <br>“hypertension-test”,<br>“test-injury_or_poisoning”, <br>“injury_or_poisoning-test”,<br>“test-kidney_disease”, <br>“kidney_disease-test”,<br>“test-obesity”, <br>“obesity-test”,<br>“test-oncological”, <br>“oncological-test”,<br>“test-psychological_condition”, <br>“psychological_condition-test”,<br>“test-symptom”, “symptom-test”,<br>“ekg_findings-disease_syndrome_disorder”,<br>“disease_syndrome_disorder-ekg_findings”,<br>“ekg_findings-heart_disease”, <br>“heart_disease-ekg_findings”,<br>“ekg_findings-symptom”, <br>“symptom-ekg_findings”,<br>“imagingfindings-cerebrovascular_disease”, <br>“cerebrovascular_disease-imagingfindings”,<br>“imagingfindings-communicable_disease”, <br>“communicable_disease-imagingfindings”,<br>“imagingfindings-disease_syndrome_disorder”, <br>“disease_syndrome_disorder-imagingfindings”,<br>“imagingfindings-heart_disease”, <br>“heart_disease-imagingfindings”,<br>“imagingfindings-hyperlipidemia”, <br>“hyperlipidemia-imagingfindings”,<br>“imagingfindings-hypertension”, <br>“hypertension-imagingfindings”,<br>“imagingfindings-injury_or_poisoning”, <br>“injury_or_poisoning-imagingfindings”,<br>“imagingfindings-kidney_disease”, <br>“kidney_disease-imagingfindings”,<br>“imagingfindings-oncological”, <br>“oncological-imagingfindings”,<br>“imagingfindings-psychological_condition”, <br>“psychological_condition-imagingfindings”,<br>“imagingfindings-symptom”, <br>“symptom-imagingfindings”,<br>“vs_finding-cerebrovascular_disease”, <br>“cerebrovascular_disease-vs_finding”,<br>“vs_finding-communicable_disease”, <br>“communicable_disease-vs_finding”,<br>“vs_finding-diabetes”, <br>“diabetes-vs_finding”,<br>“vs_finding-disease_syndrome_disorder”, <br>“disease_syndrome_disorder-vs_finding”,<br>“vs_finding-heart_disease”, <br>“heart_disease-vs_finding”,<br>“vs_finding-hyperlipidemia”, <br>“hyperlipidemia-vs_finding”,<br>“vs_finding-hypertension”, <br>“hypertension-vs_finding”,<br>“vs_finding-injury_or_poisoning”, <br>“injury_or_poisoning-vs_finding”,<br>“vs_finding-kidney_disease”, <br>“kidney_disease-vs_finding”,<br>“vs_finding-obesity”, <br>“obesity-vs_finding”,<br>“vs_finding-oncological”, <br>“oncological-vs_finding”,<br>“vs_finding-overweight”, <br>“overweight-vs_finding”,<br>“vs_finding-psychological_condition”, <br>“psychological_condition-vs_finding”,<br>“vs_finding-symptom”, “symptom-vs_finding”] |


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
ner_tagger = sparknlp.annotators.NerDLModel()\
    .pretrained('jsl_ner_wip_clinical',"en","clinical/models")\
    .setInputCols("sentences", "tokens", "embeddings")\
    .setOutputCol("ner_tags") 

re_model = RelationExtractionModel()\
    .pretrained("re_test_problem_finding", "en", 'clinical/models')\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setMaxSyntacticDistance(4)\ #default: 0
    .setPredictionThreshold(0.9)\ #default: 0.5
    .setRelationPairs(["procedure-symptom"]) # Possible relation pairs. Default: All Relations.

nlp_pipeline = Pipeline(stages=[ documenter, sentencer,tokenizer, words_embedder, pos_tagger,  clinical_ner_tagger,ner_chunker, dependency_parser,re_model])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = light_pipeline.fullAnnotate(''''Targeted biopsy of this lesion for histological correlation should be considered.'')
```

</div>

## Results

```bash
| index | relations    | entity1      | chunk1              | entity2      |  chunk2 |
|-------|--------------|--------------|---------------------|--------------|---------|
| 0     | 1            | PROCEDURE    | biopsy              | SYMPTOM      |  lesion | 
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|re_test_problem_finding|
|Type:|re|
|Compatibility:|Healthcare NLP 2.7.1+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[embeddings, pos_tags, train_ner_chunks, dependencies]|
|Output Labels:|[relations]|
|Language:|en|

## Data Source

Trained on internal datasets.