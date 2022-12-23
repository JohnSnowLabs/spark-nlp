---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 4.2.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_4_2_2
key: docs-licensed-release-notes
modify_date: 2022-11-15
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## 4.2.2

#### Highlights

+ Fine-tuning Relation Extraction models with your data
+ Added Romanian support in deidentification annotator for data obfuscation
+ New SDOH (Social Determinants of Health) ner model
+ Improved oncology models and 4 pretrained pipelines
+ New chunk mapper models to map entities (phrases) to their corresponding ICD-10-CM codes as well as clinical abbreviations to their definitions
+ New ICD-10-PCS sentence entity resolver model and ICD-10-CM resolver pipeline
+ New utility & helper modules documentation page
+ New and updated notebooks
+ 22 new clinical models and pipelines added & updated in total

</div><div class="h3-box" markdown="1">

#### Fine-Tuning Relation Extraction Models With Your Data

Instead of starting from scratch when training a new Relation Extraction model, you can train a new model by adding your new data to the pretrained model.

There are two new params in `RelationExtractionApproach` which allows you to initialize your model with the data from the pretrained model:

+ `setPretrainedModelPath`: This parameter allows you to point the training process to an existing model.
+ `setОverrideExistingLabels`: This parameter overrides the existing labels in the original model that are assigned the same output nodes in the new model. Default is True, when it is set to False the `RelationExtractionApproach` uses the existing labels and if it finds new ones it tries to assign them to unused output nodes.


*Example:*
```python
reApproach_finetune = RelationExtractionApproach()\
    .setInputCols(["embeddings", "pos_tags", "train_ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setLabelColumn("rel")\
    ...
    .setFromEntity("begin1i", "end1i", "label1")\
    .setToEntity("begin2i", "end2i", "label2")\
    .setPretrainedModelPath("existing_RE_MODEL_path")\
    .setOverrideExistingLabels(False)
```
You can check [Resume RelationExtractionApproach Training Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.4.Resume_RelationExtractionApproach_Training.ipynb) for more examples.

</div><div class="prev_ver h3-box" markdown="1">


#### Added Romanian Support in Deidentification Annotator For Data Obfuscation

Deidentification annotator is now able to obfuscate entities (coming from a deid NER model) with fake data in Romanian language.

*Example:*

```python
deid_obfuscated_faker = DeIdentification()\
    .setInputCols(["sentence", "token", "ner_chunk"]) \
    .setOutputCol("obfuscated") \
    .setMode("obfuscate")\
    .setLanguage('ro')\
    .setObfuscateDate(True)\
    .setObfuscateRefSource('faker')

text = """Nume si Prenume : BUREAN MARIA, Varsta: 77 ,Spitalul Pentru Ochi de Deal, Drumul Oprea Nr. 972 Vaslui"""
```

*Result:*

|Sentence|Masked with entity|Masked with Chars|Masked with Fixed Chars|Obfuscated|
|-|-|-|-|-|
|Nume si Prenume : BUREAN MARIA, Varsta: 77 ,Spitalul Pentru Ochi de Deal, Drumul Oprea Nr. 972 Vaslui| Nume si Prenume : <\PATIENT>, Varsta: <\AGE> ,<\HOSPITAL>, <\STREET> <\CITY> | Nume si Prenume : **********, Varsta: ** ,**************************, ****************** **** |Nume si Prenume : ****, Varsta: **** , ****, **** **** | Nume si Prenume : Claudia Crumble, Varsta: 18 ,LOS ANGELES AMBULATORY CARE CENTER, 706 north parrish avenue Piscataway|


</div><div class="prev_ver h3-box" markdown="1">


#### New SDOH (Social Determinants of Health) NER Model
 + Social Determinants of Health(SDOH) are the socioeconomic factors under which people live, learn, work, worship, and play that determine their health outcomes.The World Health Organization also provides a definition of social determinants of health. Social determinants of health as the conditions in which people are born, grow, live, work and age. These circumstances are shaped by the distribution of money, power, and resources at global, national, and local levels.  Social determinants of health (SDOH) have a major impact on people’s health, well-being, and quality of life.
  +  SDOH include lots of factors, also contribute to wide health disparities and inequities. In this project We have tried to define well these factors. The goal of this project is to train models for natural language processing focused on extracting terminology related to social determinants of health from various kinds of biomedical documents. This first model is Named Entity Recognition (NER) task.
  +  The project is still ongoing and will mature over time and the number of sdoh factors (entities) will also be enriched. It will include other tasks as well.
    
*Example:*

```python
ner_model = MedicalNerModel.pretrained("sdoh_slim_wip", "en", "clinical/models")\
        .setInputCols(["sentence", "token", "embeddings"])\
        .setOutputCol("ner")

text = """ Mother states that he does smoke, there is a family hx of alcohol on both maternal and paternal sides of the family, maternal grandfather who died of alcohol related complications and paternal grandmother with severe alcoholism. Pts own drinking began at age 16, living in LA, had a DUI at age 17 after totaling a new car that his mother bought for him, he was married. """
```

*Result:*

```bash
+-------------+-------------------+
|        token|          ner_label|
+-------------+-------------------+
|       Mother|    B-Family_Member|
|           he|           B-Gender|
|        smoke|          B-Smoking|
|      alcohol|          B-Alcohol|
|     maternal|    B-Family_Member|
|     paternal|    B-Family_Member|
|     maternal|    B-Family_Member|
|  grandfather|    B-Family_Member|
|      alcohol|          B-Alcohol|
|     paternal|    B-Family_Member|
|  grandmother|    B-Family_Member|
|       severe|          B-Alcohol|
|   alcoholism|          I-Alcohol|
|     drinking|          B-Alcohol|
|          age|              B-Age|
|           16|              I-Age|
|           LA|B-Geographic_Entity|
|          age|              B-Age|
|           17|              I-Age|
|          his|           B-Gender|
|       mother|    B-Family_Member|
|          him|           B-Gender|
|           he|           B-Gender|
|      married|   B-Marital_Status|
+-------------+-------------------+
```

</div><div class="prev_ver h3-box" markdown="1">


#### Improved Oncology NER Models And 4 New Pretrained Pipelines

We are releasing the improved version of Oncological NER models (_wip) and 4 new pretrained oncological pipelines which are able to detect assertion status and relations between the extracted oncological entities.

|NER model name (`MedicalNerModel`)| description|predicted entities|
|-|-|-|
| [ner_oncology_anatomy_general](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_anatomy_general_en.html)  | Extracting anatomical entities.  |  `Anatomical_Site`, `Direction` |
| [ner_oncology_anatomy_granular](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_anatomy_granular_en.html)  | Extracting anatomical entities using granular labels.  | `Direction`, `Site_Lymph_Node`, `Site_Breast`, `Site_Other_Body_Part`, `Site_Bone`, `Site_Liver`, `Site_Lung`, `Site_Brain`  |
| [ner_oncology_biomarker](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_biomarker_en.html)  | Extracting biomarkers and their results.  | `Biomarker`, `Biomarker_Result`  |
| [ner_oncology_demographics](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_demographics_en.html) | Extracting demographic information, including smoking status.  | `Age`, `Gender`, `Smoking_Status`, `Race_Ethnicity`  |
| [ner_oncology_diagnosis](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_diagnosis_en.html)  | Extracting entities related to cancer diagnosis, including the presence of metastasis.  | `Grade`, `Staging`, `Tumor_Size`, `Adenopathy`, `Pathology_Result`, `Histological_Type`, `Metastasis`, `Cancer_Score`, `Cancer_Dx`, `Invasion`, `Tumor_Finding`, `Performance_Status`  |
| [ner_oncology](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_en.html)  | Extracting more than 40 oncology-related entities.  | `Histological_Type`, `Direction`, `Staging`, `Cancer_Score`, `Imaging_Test`, `Cycle_Number`, `Tumor_Finding`, `Site_Lymph_Node`, `Invasion`, `Response_To_Treatment`, `Smoking_Status`, `Tumor_Size`, `Cycle_Count`, `Adenopathy`, `Age`, `Biomarker_Result`, `Unspecific_Therapy`, `Site_Breast`, `Chemotherapy`, `Targeted_Therapy`, `Radiotherapy`, `Performance_Status`, `Pathology_Test`, `Site_Other_Body_Part`, `Cancer_Surgery`, `Line_Of_Therapy`, `Pathology_Result`, `Hormonal_Therapy`, `Site_Bone`, `Biomarker`, `Immunotherapy`, `Cycle_Day`, `Frequency`, `Route`, `Duration`, `Death_Entity`, `Metastasis`, `Site_Liver`, `Cancer_Dx`, `Grade`, `Date`, `Site_Lung`, `Site_Brain`, `Relative_Date`, `Race_Ethnicity`, `Gender`, `Oncogene`, `Dosage`, `Radiation_Dose`  |
| [ner_oncology_posology](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_posology_en.html)  | This model extracts oncology specific posology information and cancer therapies.  | `Cycle_Number`, `Cycle_Count`, `Radiotherapy`, `Cancer_Surgery`, `Cycle_Day`, `Frequency`, `Route`, `Cancer_Therapy`, `Duration`, `Dosage`, `Radiation_Dose`  |
| [ner_oncology_unspecific_posology](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_unspecific_posology_en.html)  | Extracting any mention of cancer therapies and posology information using general labels  | `Cancer_Therapy`, `Posology_Information`  |
| [ner_oncology_response_to_treatment_wip](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_response_to_treatment_en.html)  | Extracting entities related to the patient's response to cancer treatment.  | `Response_To_Treatment`, `Size_Trend`, `Line_Of_Therapy`  |
| [ner_oncology_therapy](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_therapy_en.html)   |Extracting entities related to cancer therapies, including posology entities and response to treatment, using granular labels.  | `Response_To_Treatment`, `Line_Of_Therapy`, `Cancer_Surgery`, `Radiotherapy`, `Immunotherapy`, `Targeted_Therapy`, `Hormonal_Therapy`, `Chemotherapy`, `Unspecific_Therapy`, `Route`, `Duration`, `Cycle_Count`, `Dosage`, `Frequency`, `Cycle_Number`, `Cycle_Day`, `Radiation_Dose`  |
| [ner_oncology_test](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_test_en.html)  | Extracting mentions of oncology-related tests.  | `Oncogene`, `Biomarker`, `Biomarker_Result`, `Imaging_Test`, `Pathology_Test`  |
| [ner_oncology_tnm](https://nlp.johnsnowlabs.com/2022/10/25/ner_oncology_tnm_en.html)   |  Extracting mentions related to TNM staging. |  `Lymph_Node`, `Staging`, `Lymph_Node_Modifier`, `Tumor_Description`, `Tumor`, `Metastasis`, `Cancer_Dx` |


|Oncological Pipeline (`PretrainedPipeline`)| Description|
|-|-|
| [oncology_general_pipeline](https://nlp.johnsnowlabs.com/2022/11/03/oncology_general_pipeline_en.html)  | Includes Named-Entity Recognition, Assertion Status and Relation Extraction models to extract information from oncology texts. This pipeline extracts diagnoses, treatments, tests, anatomical references and demographic entities. |  
| [oncology_biomarker_pipeline](https://nlp.johnsnowlabs.com/2022/11/04/oncology_biomarker_pipeline_en.html)  | Includes Named-Entity Recognition, Assertion Status and Relation Extraction models to extract information from oncology texts. This pipeline focuses on entities related to biomarkers  |
| [oncology_diagnosis_pipeline](https://nlp.johnsnowlabs.com/2022/11/04/oncology_diagnosis_pipeline_en.html)  | Includes Named-Entity Recognition, Assertion Status, Relation Extraction and Entity Resolution models to extract information from oncology texts. This pipeline focuses on entities related to oncological diagnosis.  |
| [oncology_therapy_pipeline](https://nlp.johnsnowlabs.com/2022/11/04/oncology_therapy_pipeline_en.html)  | Includes Named-Entity Recognition and Assertion Status models to extract information from oncology texts. This pipeline focuses on entities related to therapies.  |  


*Example:*
```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("oncology_general_pipeline", "en", "clinical/models")

text = "The patient underwent a left mastectomy for a left breast cancer two months ago. The tumor is positive for ER and PR."
```
*Result:*
```bash
**** ner_oncology_wip results ****
| chunk          | ner_label        |
|:---------------|:-----------------|
| left           | Direction        |
| mastectomy     | Cancer_Surgery   |
| left           | Direction        |
| breast cancer  | Cancer_Dx        |
| two months ago | Relative_Date    |
| tumor          | Tumor_Finding    |
| positive       | Biomarker_Result |
| ER             | Biomarker        |
| PR             | Biomarker        |

**** assertion_oncology_wip results  ****
| chunk         | ner_label      | assertion   |
|:--------------|:---------------|:------------|
| mastectomy    | Cancer_Surgery | Past        |
| breast cancer | Cancer_Dx      | Present     |
| tumor         | Tumor_Finding  | Present     |
| ER            | Biomarker      | Present     |
| PR            | Biomarker      | Present     |

**** re_oncology_wip results ****
| chunk1        | entity1          | chunk2         | entity2       | relation      |
|:--------------|:-----------------|:---------------|:--------------|:--------------|
| mastectomy    | Cancer_Surgery   | two months ago | Relative_Date | is_related_to |
| breast cancer | Cancer_Dx        | two months ago | Relative_Date | is_related_to |
| tumor         | Tumor_Finding    | ER             | Biomarker     | O             |
| tumor         | Tumor_Finding    | PR             | Biomarker     | O             |
| positive      | Biomarker_Result | ER             | Biomarker     | is_related_to |
| positive      | Biomarker_Result | PR             | Biomarker     | is_related_to |
```

</div><div class="prev_ver h3-box" markdown="1">


#### New Chunk Mapper Models to Map Entities (phrases) to Their Corresponding ICD-10-CM Codes As Well As Clinical Abbreviations to Their Definitions

We have 2 new chunk mapper models:

+ `abbreviation_mapper_augmented` is an augmented version of the existing `abbreviation_mapper` model. It maps abbreviations and acronyms of medical regulatory activities to their definitions.

+ `icd10cm_mapper` maps entities to corresponding ICD-10-CM codes.

*Example:*

```python
chunkerMapper = ChunkMapperModel\
    .pretrained("icd10cm_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["icd10cm_code"])

text = """A 35-year-old male with a history of primary leiomyosarcoma of neck, gestational diabetes mellitus diagnosed eight years prior to presentation and presented with a one-week history of polydipsia, poor appetite, and vomiting."""
```

*Result:*

```bash
+------------------------------+-------+------------+
|ner_chunk                     |entity |icd10cm_code|
+------------------------------+-------+------------+
|primary leiomyosarcoma of neck|PROBLEM|C49.0       |
|gestational diabetes mellitus |PROBLEM|O24.919     |
|polydipsia                    |PROBLEM|R63.1       |
|poor appetite                 |PROBLEM|R63.0       |
|vomiting                      |PROBLEM|R11.10      |
+------------------------------+-------+------------+
```

</div><div class="prev_ver h3-box" markdown="1">



#### New ICD-10-PCS Sentence Entity Resolver Model and ICD-10-CM Resolver Pipeline

We are releasing new ICD-10-PCS resolver model and ICD-10-CM resolver pipeline:

+ `sbiobertresolve_icd10pcs_augmented` model maps extracted medical entities to ICD-10-PCS codes using `sbiobert_base_cased_mli` sentence bert embeddings. It trained on the augmented version of the dataset which is used in previous ICD-10-PCS resolver model.

*Example:*

```python
icd10pcs_resolver = SentenceEntityResolverModel\
  .pretrained("sbiobertresolve_icd10pcs_augmented","en", "clinical/models") \
  .setInputCols(["ner_chunk", "sbert_embeddings"]) \
  .setOutputCol("resolution")\
  .setDistanceFunction("EUCLIDEAN")

text = "Given the severity of her abdominal examination and her persistence of her symptoms, it is detected that need for laparoscopic appendectomy and possible open appendectomy as well as pyeloplasty. We recommend performing a mediastinoscopy"

```

*Result:*

```bash
+-------------------------+---------+-------------+------------------------------------+--------------------+
|                ner_chunk|   entity|icd10pcs_code|                         resolutions|           all_codes|
+-------------------------+---------+-------------+------------------------------------+--------------------+
|    abdominal examination|     Test|      2W63XZZ|[traction of abdominal wall [trac...|[2W63XZZ, BW40ZZZ...|
|laparoscopic appendectomy|Procedure|      0DTJ8ZZ|[resection of appendix, endo [res...|[0DTJ8ZZ, 0DT84ZZ...|
|        open appendectomy|Procedure|      0DBJ0ZZ|[excision of appendix, open appro...|[0DBJ0ZZ, 0DTJ0ZZ...|
|              pyeloplasty|Procedure|      0TS84ZZ|[reposition bilateral ureters, pe...|[0TS84ZZ, 0TS74ZZ...|
|          mediastinoscopy|Procedure|      BB1CZZZ|[fluoroscopy of mediastinum [fluo...|[BB1CZZZ, 0WJC4ZZ...|
+-------------------------+---------+-------------+------------------------------------+--------------------+
```

+ `icd10cm_resolver_pipeline` pretrained pipeline maps entities with their corresponding ICD-10-CM codes. You’ll just feed your text and it will return the corresponding ICD-10-CM codes.

*Example:*

```python
from sparknlp.pretrained import PretrainedPipeline

resolver_pipeline = PretrainedPipeline("icd10cm_resolver_pipeline", "en", "clinical/models")

text = "A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years and anisakiasis. Also, it was reported that fetal and neonatal hemorrhage"
```

*Result:*

```bash
+-----------------------------+---------+------------+
|chunk                        |ner_chunk|icd10cm_code|
+-----------------------------+---------+------------+
|gestational diabetes mellitus|PROBLEM  |O24.919     |
|anisakiasis                  |PROBLEM  |B81.0       |
|fetal and neonatal hemorrhage|PROBLEM  |P545        |
+-----------------------------+---------+------------+
```


</div><div class="prev_ver h3-box" markdown="1">

#### New Utility & Helper Modules Documentation Page

We have a new [utility & helper modules documentation page](https://nlp.johnsnowlabs.com/docs/en/utility_helper_modules) that you can find the documentations of Spark NLP for Healthcare modules with examples.


</div><div class="prev_ver h3-box" markdown="1">


#### New and Updated Notebooks

+ New [Resume RelationExtractionApproach Training](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.4.Resume_RelationExtractionApproach_Training.ipynb) notebook train a model already trained on a different dataset.

+ Updated [Clinical Deidentification](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb) notebook with day shifting feature in `DeIdentification`.

+ Updated [Clinical Multi Language Deidentification](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.1.Clinical_Multi_Language_Deidentification.ipynb) notebook with new Romanian obfuscation and faker improvement.

+ Updated [Adverse Drug Event ADE NER and Classifier](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb) notebook with the new models and improvement.

</div><div class="prev_ver h3-box" markdown="1">


#### 22 New Clinical Models and Pipelines Added & Updated in Total


+ `abbreviation_mapper_augmented`
+ `icd10cm_mapper`
+ `sbiobertresolve_icd10pcs_augmented`
+ `icd10cm_resolver_pipeline`
+ `oncology_biomarker_pipeline`
+ `oncology_diagnosis_pipeline`
+ `oncology_therapy_pipeline`
+ `oncology_general_pipeline`
+ `ner_oncology_anatomy_general`
+ `ner_oncology_anatomy_granular`
+ `ner_oncology_biomarker`
+ `ner_oncology_demographics`
+ `ner_oncology_diagnosis`
+ `ner_oncology`
+ `ner_oncology_posology`
+ `ner_oncology_response_to_treatment`
+ `ner_oncology_test`
+ `ner_oncology_therapy`
+ `ner_oncology_tnm`
+ `ner_oncology_unspecific_posology`
+ `sdoh_slim_wip`
+ `t5_base_pubmedqa`

For all Spark NLP for healthcare models, please check: [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Healthcare+NLP)


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

{%- include docs-healthcare-pagination.html -%}