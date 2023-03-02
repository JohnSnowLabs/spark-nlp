---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes
permalink: /docs/en/spark_nlp_healthcare_versions/licensed_release_notes
key: docs-licensed-release-notes
modify_date: 2023-03-02
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## 4.3.1

#### Highlights

+ The first Voice of Patients (VOP) named entity recognition model
+ New Social Determinants of Health (SDOH) named entity recognition models
+ New entity resolution model for mapping Rxnorm codes according to the National Institute of Health (NIH) Database
+ New Chunk Mapper models for mapping NDC codes to drug brand names as well as clinical entities (like drugs/ingredients) to Rxnorm codes
+ Format consistency for formatted entity obfuscation in `Deidentification` module
+ New parameters for controlling the validation set while training a NER model with `MedicalNerApproach`
+ Whitelisting the entities while merging multiple entities in `ChunkMergeApproach`
+ Core improvements and bug fixes
+ New and updated notebooks
+ New and updated demos
+ 8 new clinical models and pipelines added & updated in total

</div><div class="h3-box" markdown="1">

#### The First Voice of Patients (VOP) Named Entity Recognition Model

We are releasing a new VOP NER model that was trained on the conversations gathered from patients forums.


| model name                                     | description                                                                                         | predicted entities                     |
|------------------------------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------|
| [ner_vop_slim_wip](https://nlp.johnsnowlabs.com/2023/02/25/ner_vop_slim_wip_en.html) | This model extracts healthcare-related terms from the documents transferred from the patient's own sentences. | `AdmissionDischarge` `Age` `BodyPart` `ClinicalDept` `DateTime` `Disease` `Dosage_Strength` `Drug` `Duration` `Employment` `Form` `Frequency` `Gender` `Laterality` `Procedure` `PsychologicalCondition` `RelationshipStatus` `Route` `Symptom` `Test` `Vaccine` `VitalTest` |


*Example*:

```python
...
clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_vop_slim_wip", "en", "clinical/models")\
    .setInputCols(["sentence", "token","embeddings"])\
    .setOutputCol("ner")

sample_texts = ["Hello,I'm 20 year old girl. I'm diagnosed with hyperthyroid 1 month ago. I was feeling weak, poor digestion, depression, left chest pain, increased heart rate from 4 months. Also i have b12 deficiency so I'm taking weekly supplement of 1000 mcg b12 daily."]
```

*Result*:

```bash
+--------------------+-----+----+----------------------+
|chunk               |begin|end |ner_label             |
+--------------------+-----+----+----------------------+
|20 year old         |10   |20  |Age                   |
|girl                |22   |25  |Gender                |
|hyperthyroid        |47   |58  |Disease               |
|1 month ago         |60   |70  |DateTime              |
|weak                |87   |90  |Symptom               |
|depression          |137  |146 |PsychologicalCondition|
|left                |149  |152 |Laterality            |
|chest               |154  |158 |BodyPart              |
|pain                |160  |163 |Symptom               |
|heart rate          |176  |185 |VitalTest             |
|4 months            |215  |222 |Duration              |
|b12 deficiency      |613  |626 |Disease               |
|weekly              |667  |672 |Frequency             |
|supplement          |674  |683 |Drug                  |
|1000 mcg            |702  |709 |Dosage_Strength       |
|b12                 |711  |713 |Drug                  |
|daily               |715  |719 |Frequency             |
+--------------------+-----+----+----------------------+
```

</div><div class="h3-box" markdown="1">

#### New Social Determinants of Health (SDOH) Named Entity Recognition Models

We are releasing 4 new SDOH NER models with various entity combinations.

| model name                                     | description                                                                                         | predicted entities                     |
|------------------------------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------|
| [ner_sdoh_substance_usage_wip](https://nlp.johnsnowlabs.com/2023/02/23/ner_sdoh_substance_usage_wip_en.html)     | This model extracts substance usage information related to Social Determinants of Health from various kinds of biomedical documents.     | `Smoking` `Substance_Duration` `Substance_Use` `Substance_Quantity`   `Substance_Frequency` `Alcohol`    |
| [ner_sdoh_access_to_healthcare_wip](https://nlp.johnsnowlabs.com/2023/02/24/ner_sdoh_access_to_healthcare_wip_en.html)                 | This model extracts access to healthcare information related to Social Determinants of Health from various kinds of biomedical documents.              | `Insurance_Status` `Healthcare_Institution` `Access_To_Care`  |
| [ner_sdoh_community_condition_wip](https://nlp.johnsnowlabs.com/2023/02/24/ner_sdoh_community_condition_wip_en.html) | This model extracts community condition information related to Social Determinants of Health from various kinds of biomedical documents. | `Transportation` `Community_Living_Conditions` `Housing` `Food_Insecurity` |
| [ner_sdoh_health_behaviours_problems_wip](https://nlp.johnsnowlabs.com/2023/02/24/ner_sdoh_health_behaviours_problems_wip_en.html) | This model extracts health and behaviours problems related to Social Determinants of Health from various kinds of biomedical documents. | `Diet` `Mental_Health` `Obesity` `Eating_Disorder` `Sexual_Activity` `Disability` `Quality_Of_Life` `Other_Disease`  `Exercise` `Communicable_Disease` `Hyperlipidemia` `Hypertension` |


- ner_sdoh_substance_usage_wip

*Example*:

```python
...
clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_sdoh_substance_usage_wip", "en", "clinical/models")\
    .setInputCols(["sentence", "token","embeddings"])\
    .setOutputCol("ner")

sample_texts = ["He does drink occasional alcohol approximately 5 to 6 alcoholic drinks per month.",
"He continues to smoke one pack of cigarettes daily, as he has for the past 28 years."]
```

*Result*:

```bash
+----------------+-----+---+-------------------+
|chunk           |begin|end|ner_label          |
+----------------+-----+---+-------------------+
|drink           |8    |12 |Alcohol            |
|occasional      |14   |23 |Substance_Frequency|
|alcohol         |25   |31 |Alcohol            |
|5 to 6          |47   |52 |Substance_Quantity |
|alcoholic drinks|54   |69 |Alcohol            |
|per month       |71   |79 |Substance_Frequency|
|smoke           |16   |20 |Smoking            |
|one pack        |22   |29 |Substance_Quantity |
|cigarettes      |34   |43 |Smoking            |
|daily           |45   |49 |Substance_Frequency|
|past 28 years   |70   |82 |Substance_Duration |
+----------------+-----+---+-------------------+
```


- ner_sdoh_access_to_healthcare_wip

*Example*:

```python
...
sample_texts = ["She has a pension and private health insurance, she reports feeling lonely and isolated.",
               "He also reported food insecurityduring his childhood and lack of access to adequate healthcare.",
               "She used to work as a unit clerk at XYZ Medical Center."]

```

*Result*:

```bash
+-----------------------------+-----+---+----------------------+
|chunk                        |begin|end|ner_label             |
+-----------------------------+-----+---+----------------------+
|private health insurance     |22   |45 |Insurance_Status      |
|access to adequate healthcare|65   |93 |Access_To_Care        |
|XYZ Medical Center           |36   |53 |Healthcare_Institution|
+-----------------------------+-----+---+----------------------+
```

- ner_sdoh_community_condition_wip

*Example*:

```python
...
sample_texts = ["He is currently experiencing financial stress due to job insecurity, and he lives in a small apartment in a densely populated area with limited access to green spaces and outdoor recreational activities.",
               "Patient reports difficulty affording healthy food, and relies oncheaper, processed options.",
               "She reports her husband and sons provide transportation top medical apptsand do her grocery shopping."]
```

*Result*:

```bash
+-------------------------------+-----+---+---------------------------+
|chunk                          |begin|end|ner_label                  |
+-------------------------------+-----+---+---------------------------+
|small apartment                |87   |101|Housing                    |
|green spaces                   |154  |165|Community_Living_Conditions|
|outdoor recreational activities|171  |201|Community_Living_Conditions|
|healthy food                   |37   |48 |Food_Insecurity            |
|transportation                 |41   |54 |Transportation             |
+-------------------------------+-----+---+---------------------------+
```

- ner_sdoh_health_behaviours_problems_wip

*Example*:

```python
...

sample_texts = ["She has not been getting regular exercise and not followed diet for approximately two years due to chronic sciatic pain.",
               "Medical History: The patient is a 32-year-old female who presents with a history of anxiety, depression, bulimia nervosa, elevated cholesterol, and substance abuse.",
               "Pt was intubated atthe scene & currently sedated due to high BP. Also, he is currently on social security disability."]
```

*Result*:

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

</div><div class="h3-box" markdown="1">

#### New Entity Resolver Model for Mapping Rxnorm Codes According To the National Institute of Health (NIH) Database

We are releasing `sbiobertresolve_rxnorm_nih` pretrained model to map clinical entities and concepts (like drugs/ingredients) to RxNorm codes according to the National Institute of Health (NIH) database using `sbiobert_base_cased_mli` Sentence Bert Embeddings.

*Example*:

```python
...
rxnorm_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_rxnorm_nih","en", "clinical/models") \
     .setInputCols(["sbert_embeddings"]) \
     .setOutputCol("resolution")\
     .setDistanceFunction("EUCLIDEAN")

text= "She is given folic acid 1 mg daily , levothyroxine 0.1 mg and aspirin 81 mg daily ."
```


*Result:*

```bash
| ner_chunk            | entity |rxnorm_code | all_codes                               | resolutions                                                                      |
|:---------------------|:-------|-----------:|:----------------------------------------|:---------------------------------------------------------------------------------|
| folic acid 1 mg      | DRUG   |   12281181 | ['12281181', '12283696', '12270292', ...| ['folic acid 1 MG [folic acid 1 MG]', 'folic acid 1.1 MG [folic acid 1.1 MG]',...|
| levothyroxine 0.1 mg | DRUG   |   12275630 | ['12275630', '12275646', '12301585', ...| ['levothyroxine sodium 0.1 MG [levothyroxine sodium 0.1 MG]', 'levothyroxine  ...|
| aspirin 81 mg        | DRUG   |   12278696 | ['12278696', '12299811', '12298729', ...| ['aspirin 81 MG [aspirin 81 MG]', 'aspirin 81 MG [YSP Aspirin] [aspirin 81 MG ...|
```

</div><div class="h3-box" markdown="1">

#### New Chunk Mapper Models For Mapping NDC Codes to Drug Brand Names As Well As Clinical Entities (like drugs/ingredients) to Rxnorm Codes

 We have two new chunk mapper models.

 + `ndc_drug_brandname_mapper` model maps NDC codes with their corresponding drug brand names as well as RxNorm Codes According to According to National Institute of Health (NIH).

*Example*:

```python
...
mapper = ChunkMapperModel.pretrained("ndc_drug_brandname_mapper", "en", "clinical/models")\
    .setInputCols("document")\
    .setOutputCol("mappings")\
    .setRels(["drug_brand_name"])\

text= ["0009-4992", "57894-150"]
```

 *Result:*

```bash
|    | ndc_code   | drug_brand_name   |
|---:|:-----------|:------------------|
|  0 | 0009-4992  | ZYVOX             |
|  1 | 57894-150  | ZYTIGA            |
```

+ `rxnorm_nih_mapper` model maps entities with their corresponding RxNorm codes according to the National Institute of Health (NIH) database. It returns Rxnorm codes along with their NIH Rxnorm Term Types within a parenthesis.

*Example*:

```python
...
chunkerMapper = ChunkMapperModel\
 .pretrained("rxnorm_nih_mapper", "en", "clinical/models")\
 .setInputCols(["ner_chunk"])\
 .setOutputCol("mappings")\
 .setRels(["rxnorm_code"])
```

 *Result*:

```bash
+-------------------------+-------------+-----------+
|ner_chunk                |mappings     |relation   |
+-------------------------+-------------+-----------+
|Adapin 10 MG Oral Capsule|1911002 (SY) |rxnorm_code|
|acetohexamide            |12250421 (IN)|rxnorm_code|
|Parlodel                 |829 (BN)     |rxnorm_code|
+-------------------------+-------------+-----------+
```

</div><div class="h3-box" markdown="1">

#### Format Consistency For Formatted Entity Obfuscation In `Deidentification` Module

We have added a new `setSameLengthFormattedEntities` parameter that obfuscates the formatted entities like `PHONE`, `FAX`, `ID`, `IDNUM`, `BIOID`, `MEDICALRECORD`, `ZIP`, `VIN`, `SSN`, `DLN`, `PLATE` and `LICENSE` with the fake ones in the same format. Default is an empty list (`[]`).

*Example*:

```python
obfuscated = DeIdentification()\
    .setInputCols(["sentence", "token", "deid_ner_chunk"]) \
    .setOutputCol("obfuscated") \
    .setMode("obfuscate")\
    .setLanguage('en')\
    .setObfuscateDate(True)\
    .setObfuscateRefSource('faker')\
    .setSameLengthFormattedEntities(["PHONE","MEDICALRECORD", "IDNUM"])

sample_text = """Record date: 2003-01-13
Name : Hendrickson, Ora, Age: 25
MR: #7194334
ID: 1231511863
Phone: (302) 786-5227"""
```

*Result*:

```bash
+--------------------------------+--------------------------------+------------------------------+
|                        sentence|                         masking|                   obfuscation|
+-------------------------------:+-------------------------------:+-----------------------------:+
|         Record date: 2003-01-13|           Record date: \<DATE\>|       Record date: 2003-03-07|
|Name : Hendrickson, Ora, Age: 25|Name : \<PATIENT\>, Age: \<AGE\>|Name : Manya Horsfall, Age: 20|
|                   MR:  #7194334|           MR: \<MEDICALRECORD\>|                  MR: #4868080|
|                  ID: 1231511863|                   ID: \<IDNUM\>|                ID: 2174658035|
|           Phone: (302) 786-5227|                 Phone:\<PHONE\>|         Phone: (467) 302-9509|
+--------------------------------+--------------------------------+------------------------------+
```

</div><div class="h3-box" markdown="1">

#### New Parameters For Controlling The Validation Set While Training a NER Model With `MedicalNerApproach`

We added a new parameter to `MedicalNerApproach` for controlling the validation set while training.

+ `setRandomValidationSplitPerEpoch`: If it is `True`, the validation set is randomly splitted for each epoch; and if it is `False`, the split is done only once before training (the same validation split used after each epoch). Default is `False`.

*Example*:

```python
nerTagger = MedicalNerApproach()\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setLabelColumn("label")\
    .setValidationSplit(0.2)\
    .setRandomValidationSplitPerEpoch(True)\
    .setRandomSeed(42)\
    ...
```

</div><div class="h3-box" markdown="1">

#### Whitelisting The Entities While Merging Multiple Entities In `ChunkMergeApproach`

We have added `setWhiteList` parameter to `ChunkMergeApproach` annotator that you can whitelist detected entities while merging.

*Example*:

```python
chunk_merge = ChunkMergeApproach()\
      .setInputCols("deid_chunk_1", "deid_chunk_2")\
      .setOutputCol("merged_chunk")\
      .setMergeOverlapping(True)\
      #.setWhiteList(["AGE","DATE"])

sample_text = "Mr. ABC is a 25 years old with a nonproductive cough that started last week. He has a history of pericarditis in May 2006 and developed cough with right-sided chest pain, and admitted to Beverley Count Hospital."
```

*Result for without `WhiteList`*:

```bash
| index | ner_chunk               | entity   |
|-------|-------------------------|----------|
| 0     | John Smith              | PATIENT  |
| 1     | 25                      | AGE      |
| 2     | May 2006                | DATE     |
| 3     | Beverley Count Hospital | HOSPITAL |
```

*Result for with `WhiteList(["AGE","DATE"])`*:

```bash
| index | ner_chunk               | entity   |
|-------|-------------------------|----------|
| 0     | 25                      | AGE      |
| 1     | May 2006                | DATE     |
```

</div><div class="h3-box" markdown="1">

#### Core Improvements and Bug Fixes

- Fixed the bug in `get_assertion_data` method issue in ALAB module
- Updated documentation pages with corrections and additions.

</div><div class="h3-box" markdown="1">

#### New and Updated Notebooks

- Updated [Spark NLP for Healthcare Workshop in 3 hr](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/healthcare-nlp/00.SparkNLP_for_Healthcare_3h_Notebook.ipynb) with latest examples.

</div><div class="h3-box" markdown="1">

#### New and Updated Demos

+ [SOCIAL_DETERMINANT_ALCOHOL](https://demo.johnsnowlabs.com/healthcare/SOCIAL_DETERMINANT_ALCOHOL/ ) demo
+ [SOCIAL_DETERMINANT_TOBACCO](https://demo.johnsnowlabs.com/healthcare/SOCIAL_DETERMINANT_TOBACCO/ ) demo

</div><div class="h3-box" markdown="1">

#### 8 New Clinical Models and Pipelines Added & Updated in Total


+ `ner_sdoh_substance_usage_wip`
+ `ner_sdoh_access_to_healthcare_wip`
+ `ner_sdoh_community_condition_wip`
+ `ner_sdoh_health_behaviours_problems_wip`
+ `ner_vop_slim_wip`
+ `sbiobertresolve_rxnorm_nih`
+ `ndc_drug_brandname_mapper`
+ `rxnorm_nih_mapper`

</div><div class="h3-box" markdown="1">

For all Spark NLP for Healthcare models, please check: [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Healthcare+NLP)

</div>
<div class="prev_ver h3-box" markdown="1">

## Previous versions

</div>
{%- include docs-healthcare-pagination.html -%}
