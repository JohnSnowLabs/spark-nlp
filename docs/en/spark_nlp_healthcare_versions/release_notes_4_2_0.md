---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 4.2.0
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_4_2_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## 4.2.0

#### Highlights

+ Introducing 46 new Oncology specific pretrained models (12 NER, 12 BERT-based token classification, 14 relation extraction, 8 assertion status models)
+ Brand new `NerQuestionGenerator` annotator for automated prompt generation for a QA-based Zero-Shot NER model
+ Updated ALAB (Annotation Lab) module becoming a fullfledged suite to manage activities on ALAB via its API remotely
+ New pretrained assertion status detection model (`assertion_jsl_augmented`) to classify the negativity & assertion scope of medical concepts
+ New chunk mapper models and pretrained pipeline to map entities (phrases) to their corresponding ICD-9, ICD-10-CM and RxNorm codes
+ New ICD-9-CM sentence entity resolver model and pretrained pipeline
+ New shifting days feature in `DeIdentification` by using the new `DocumentHashCoder` annotator
+ Updated NER model finder pretrained pipeline to help users find the most appropriate NER model for their use case in one-liner
+ Medicare risk adjustment score calculation module updated to support different version and year combinations
+ Core improvements and bug fixes
+ New and updated notebooks
+ 50+ new clinical models and pipelines added & updated in total

</div><div class="h3-box" markdown="1">

#### Introducing 46 New Oncology Specific Pretrained Models (12 NER, 12 BERT-Based Token Classification, 14 Relation Extraction, 8 Assertion Status Models)

These models will be the first versions (wip - work in progress) of Oncology models.

</div><div class="h3-box" markdown="1">

##### New Oncological NER and BERT-Based Token Classification Models

We have 12 new oncological NER and their BERT-based token classification models.

|NER model name (MedicalNerModel)|BERT-Based model name (MedicalBertForTokenClassifier)| description|predicted entities|
|-|-|-|-|
| [ner_oncology_therapy_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_therapy_wip_en.html)  |bert_token_classifier_ner_oncology_therapy_wip |This model extracts entities related to cancer therapies, including posology entities and response to treatment, using granular labels.  | `Response_To_Treatment`, `Line_Of_Therapy`, `Cancer_Surgery`, `Radiotherapy`, `Immunotherapy`, `Targeted_Therapy`, `Hormonal_Therapy`, `Chemotherapy`, `Unspecific_Therapy`, `Route`, `Duration`, `Cycle_Count`, `Dosage`, `Frequency`, `Cycle_Number`, `Cycle_Day`, `Radiation_Dose`  |
| [ner_oncology_diagnosis_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_diagnosis_wip_en.html)  | bert_token_classifier_ner_oncology_diagnosis_wip |This model extracts entities related to cancer diagnosis, including the presence of metastasis.  | `Grade`, `Staging`, `Tumor_Size`, `Adenopathy`, `Pathology_Result`, `Histological_Type`, `Metastasis`, `Cancer_Score`, `Cancer_Dx`, `Invasion`, `Tumor_Finding`, `Performance_Status`  |
| [ner_oncology_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_wip_en.html)  | bert_token_classifier_ner_oncology_wip | This model extracts more than 40 oncology-related entities.  | `Histological_Type`, `Direction`, `Staging`, `Cancer_Score`, `Imaging_Test`, `Cycle_Number`, `Tumor_Finding`, `Site_Lymph_Node`, `Invasion`, `Response_To_Treatment`, `Smoking_Status`, `Tumor_Size`, `Cycle_Count`, `Adenopathy`, `Age`, `Biomarker_Result`, `Unspecific_Therapy`, `Site_Breast`, `Chemotherapy`, `Targeted_Therapy`, `Radiotherapy`, `Performance_Status`, `Pathology_Test`, `Site_Other_Body_Part`, `Cancer_Surgery`, `Line_Of_Therapy`, `Pathology_Result`, `Hormonal_Therapy`, `Site_Bone`, `Biomarker`, `Immunotherapy`, `Cycle_Day`, `Frequency`, `Route`, `Duration`, `Death_Entity`, `Metastasis`, `Site_Liver`, `Cancer_Dx`, `Grade`, `Date`, `Site_Lung`, `Site_Brain`, `Relative_Date`, `Race_Ethnicity`, `Gender`, `Oncogene`, `Dosage`, `Radiation_Dose`  |
| [ner_oncology_tnm_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_tnm_wip_en.html)  | bert_token_classifier_ner_oncology_tnm_wip |  This model extracts mentions related to TNM staging. |  `Lymph_Node`, `Staging`, `Lymph_Node_Modifier`, `Tumor_Description`, `Tumor`, `Metastasis`, `Cancer_Dx` |
| [ner_oncology_anatomy_general_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_anatomy_general_wip_en.html)  | bert_token_classifier_ner_oncology_anatomy_general_wip| This model extracts anatomical entities.  |  `Anatomical_Site`, `Direction` |
| [ner_oncology_demographics_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_demographics_wip_en.html)  | bert_token_classifier_ner_oncology_demographics_wip| This model extracts demographic information, including smoking status.  | `Age`, `Gender`, `Smoking_Status`, `Race_Ethnicity`  |
| [ner_oncology_test_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_test_wip_en.html)  | bert_token_classifier_ner_oncology_test_wip| This model extracts mentions of oncology-related tests.  | `Oncogene`, `Biomarker`, `Biomarker_Result`, `Imaging_Test`, `Pathology_Test`  |
| [ner_oncology_unspecific_posology_wip](https://nlp.johnsnowlabs.com/2022/09/30/ner_oncology_unspecific_posology_wip_en.html)  |bert_token_classifier_ner_oncology_unspecific_posology_wip| This model extracts any mention of cancer therapies and posology information using general labels  | `Cancer_Therapy`, `Posology_Information`  |
| [ner_oncology_anatomy_granular_wip](https://nlp.johnsnowlabs.com/2022/10/01/ner_oncology_anatomy_granular_wip_en.html)  |bert_token_classifier_ner_oncology_anatomy_granular_wip| This model extracts anatomical entities using granular labels.  | Direction, Site_Lymph_Node, Site_Breast, Site_Other_Body_Part, Site_Bone, Site_Liver, Site_Lung, Site_Brain  |
| [ner_oncology_response_to_treatment_wip](https://nlp.johnsnowlabs.com/2022/10/01/ner_oncology_response_to_treatment_wip_en.html)  |bert_token_classifier_ner_oncology_response_to_treatment_wip| This model extracts entities related to the patient's response to cancer treatment.  | `Response_To_Treatment`, `Size_Trend`, `Line_Of_Therapy`  |
| [ner_oncology_biomarker_wip](https://nlp.johnsnowlabs.com/2022/10/01/ner_oncology_biomarker_wip_en.html)  | bert_token_classifier_ner_oncology_biomarker_wip| This model extracts biomarkers and their results.  | `Biomarker`, `Biomarker_Result`  |
| [ner_oncology_posology_wip](https://nlp.johnsnowlabs.com/2022/10/01/ner_oncology_posology_wip_en.html)  | bert_token_classifier_ner_oncology_posology_wip| This model extracts oncology specific posology information and cancer therapies.  | `Cycle_Number`, `Cycle_Count`, `Radiotherapy`, `Cancer_Surgery`, `Cycle_Day`, `Frequency`, `Route`, `Cancer_Therapy`, `Duration`, `Dosage`, `Radiation_Dose`  |

**F1 Scores:**

|label|f1|label|f1|label|f1|label|f1|label|f1|
|---|---|---|---|---|---|---|---|---|---|
|Adenopathy|0\.73|Cycle\_Day|0\.83|Histological\_Type|0\.71|Posology\_Information|0\.88|Site\_Lymph\_Node|0\.91|
|Age|0\.97|Cycle\_Number|0\.79|Hormonal\_Therapy|0\.90|Race\_Ethnicity|0\.86|Smoking\_Status|0\.82|
|Anatomical\_Site|0\.83|Date|0\.97|Imaging\_Test|0\.90|Radiation\_Dose|0\.87|Staging|0\.85|
|Biomarker|0\.89|Death\_Entity|0\.82|Invasion|0\.80|Radiotherapy|0\.90|Targeted\_Therapy|0\.87|
|Biomarker\_Result|0\.82|Direction|0\.82|Line\_Of\_Therapy|0\.91|Relative\_Date|0\.79|Tumor|0\.91|
|Cancer\_Dx|0\.92|Dosage|0\.91|Lymph\_Node|0\.86|Route|0\.84|Tumor\_Description|0\.81|
|Cancer\_Surgery|0\.85|Duration|0\.77|Lymph\_Node\_Modifier|0\.75|Site\_Bone|0\.80|Tumor\_Finding|0\.92|
|Cancer\_Therapy|0\.90|Frequency|0\.88|Metastasis|0\.95|Site\_Brain|0\.78|Tumor\_Size|0\.88|
|Chemotherapy|0\.90|Gender|0\.99|Oncogene|0\.77|Site\_Breast|0\.88|
|Cycle\_Count|0\.81|Grade|0\.81|Pathology\_Test|0\.79|Site\_Lung|0\.79|



*NER Model Example:*

```python
...
medical_ner = MedicalNerModel.pretrained("ner_oncology_wip", "en", "clinical/models") \
                   .setInputCols(["sentence", "token", "embeddings"]) \
                   .setOutputCol("ner")
...

sample_text = "The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago. The tumor was positive for ER. Postoperatively, radiotherapy was administered to her breast."
```

*BERT-Based Token Classification Model Example:*

```python
...
tokenClassifier = MedicalBertForTokenClassifier.pretrained("bert_token_classifier_ner_oncology_wip", "en", "clinical/models")\
    .setInputCols("token", "document")\
    .setOutputCol("ner")\
    .setCaseSensitive(True)
...

sample_text = "The had previously undergone a left mastectomy and an axillary lymph node dissection for a left breast cancer twenty years ago. The tumor was positive for ER. Postoperatively, radiotherapy was administered to her breast."
```

*Results:*

```bash
+------------------------------+---------------------+
|chunk                         |ner_label            |
+------------------------------+---------------------+
|left                          |Direction            |
|mastectomy                    |Cancer_Surgery       |
|axillary lymph node dissection|Cancer_Surgery       |
|left                          |Direction            |
|breast cancer                 |Cancer_Dx            |
|twenty years ago              |Relative_Date        |
|tumor                         |Tumor_Finding        |
|positive                      |Biomarker_Result     |
|ER                            |Biomarker            |
|radiotherapy                  |Radiotherapy         |
|her                           |Gender               |
|breast                        |Site_Breast          |
+------------------------------+---------------------+
```

</div><div class="h3-box" markdown="1">

##### New Oncological  Assertion Status Models

We have 8 new oncological assertion status detection models.

|model name|description|predicted entities|
|-|-|-|
|assertion_oncology_wip   | This model identifies the assertion status of different oncology-related entities.  | `Medical_History`, `Family_History`, `Possible`, `Hypothetical_Or_Absent`  |
| assertion_oncology_problem_wip  |This assertion model identifies the status of Cancer_Dx extractions and other problem entities.   | `Present`, `Possible`, `Hypothetical`, `Absent`, `Family`  |
| assertion_oncology_treatment_wip  | This model identifies the assertion status of treatments mentioned in text.  | `Present`, `Planned`, `Past`, `Hypothetical`, `Absent`  |
| assertion_oncology_response_to_treatment_wip  | This assertion model identifies if the response to treatment mentioned in text actually happened, or if it mentioned as something absent or hypothetical.  | `Present_Or_Past`, `Hypothetical_Or_Absent`  |
| assertion_oncology_test_binary_wip  | This assertion model identifies if a test mentioned in text actually was used, or if it mentioned as something absent or hypothetical.  | `Present_Or_Past`, `Hypothetical_Or_Absent`  |
| assertion_oncology_smoking_status_wip  | This assertion model is used to classify the smoking status of the patient.  |`Absent`, `Past`, `Present`   |
| assertion_oncology_family_history_wip  | This assertion model identifies if an entity refers to a family member.  | `Family_History`, `Other`  |
| assertion_oncology_demographic_binary_wip  | This assertion model identifies if the demographic entities refer to the patient or to someone else.  | `Patient`, `Someone_Else`  |

*Example:*

```python
...
assertion = AssertionDLModel.pretrained("assertion_oncology_problem_wip", "en", "clinical/models") \
                .setInputCols(["sentence", 'ner_chunk', "embeddings"]) \
                .setOutputCol("assertion")
...

sample_text = "Considering the findings, the patient may have a breast cancer. There are no signs of metastasis. Family history positive for breast cancer in her maternal grandmother."
```

*Results:*

```bash
+-------------+----------+---------+
|        chunk| ner_label|assertion|
+-------------+----------+---------+
|breast cancer| Cancer_Dx| Possible|
|   metastasis|Metastasis|   Absent|
|breast cancer| Cancer_Dx|   Family|
+-------------+----------+---------+
```

</div><div class="h3-box" markdown="1">

##### New Oncological Relation Extraction Models
We are releasing 7 new `RelationExtractionModel` and 7 new `RelationExtractionDLModel` models to extract relations between various oncological concepts.

| model name                                          	| description                                                                	| predicted entities          	|
|-------------------------------------------------------|-----------------------------------------------------------------------------|-------------------------------|
| [re_oncology_size_wip](https://nlp.johnsnowlabs.com/2022/09/26/re_oncology_size_wip_en.html)                          | This model links Tumor_Size extractions to their corresponding Tumor_Finding extractions.   	          | `is_size_of`, `O`   	|
| [re_oncology_biomarker_result_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_biomarker_result_wip_en.html)  | This model links Biomarker and Oncogene extractions to their corresponding Biomarker_Result extractions.| `is_finding_of`, `O`	|
| [re_oncology_granular_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_granular_wip_en.html)                  | This model can be identified four relation types                                                        | `is_size_of`, `is_finding_of`, `is_date_of`, `is_location_of`, `O` |
| [re_oncology_location_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_location_wip_en.html)                  | This model links extractions from anatomical entities (such as Site_Breast or Site_Lung) to other clinical entities (such as Tumor_Finding or Cancer_Surgery).| `is_location_of`, `O`|
| [re_oncology_temporal_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_temporal_wip_en.html) 	                | This model links Date and Relative_Date extractions to clinical entities such as Test or Cancer_Dx.| `is_date_of`, `O`|
| [re_oncology_test_result_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_test_result_wip_en.html)            | This model links test extractions to their corresponding results.| `is_finding_of`, `O`|
| [re_oncology_wip](https://nlp.johnsnowlabs.com/2022/09/27/re_oncology_wip_en.html)  | This model link between dates and other clinical entities, between tumor mentions and their size, between anatomical entities and other clinical entities, and between tests and their results.|`is_related_to`, `O`|
| [redl_oncology_size_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/28/redl_oncology_size_biobert_wip_en.html)                        | This model links Tumor_Size extractions to their corresponding Tumor_Finding extractions.                  | `is_size_of`, `O` |
| [redl_oncology_biomarker_result_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_biomarker_result_biobert_wip_en.html)| This model links Biomarker and Oncogene extractions to their corresponding Biomarker_Result extractions.   | `is_finding_of`, `O` |
| [redl_oncology_location_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_location_biobert_wip_en.html)                |This model links extractions from anatomical entities (such as Site_Breast or Site_Lung) to other clinical entities (such as Tumor_Finding or Cancer_Surgery).  | `is_location_of`, `O` |
| [redl_oncology_temporal_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_temporal_biobert_wip_en.html)                |This model links Date and Relative_Date extractions to clinical entities such as Test or Cancer_Dx.         | `is_date_of`, `O` |
| [redl_oncology_test_result_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_test_result_biobert_wip_en.html)      |This model links test extractions to their corresponding results.                                           | `is_finding_of`, `O` |
| [redl_oncology_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_biobert_wip_en.html)                                  |This model identifies relations between dates and other clinical entities, between tumor mentions and their size, between anatomical entities and other clinical entities, and between tests and their results.|`is_related_to` |
| [redl_oncology_granular_biobert_wip](https://nlp.johnsnowlabs.com/2022/09/29/redl_oncology_granular_biobert_wip_en.html)                |This model can be identified four relation types                                            | `is_date_of`, `is_finding_of`, `is_location_of`, `is_size_of`, `O` |

**F1 Scores and Samples:**

|label|F1 Score|sample_text|results|
|-|-|-|-|
|is_finding_of   | 0.95  |"Immunohistochemistry was negative for thyroid transcription factor-1 and napsin A."|`negative - thyroid transcription factor-1`, `negative - napsin`|
|is_date_of   | 0.81  |"A mastectomy was performed two months ago."|`mastectomy-two months ago`|
|is_location_of   | 0.92  |"In April 2011, she first noticed a lump in her right breast."|`lump - breast`|
|is_size_of   | 0.86  |"The patient presented a 2 cm mass in her left breast."|`2 cm - mass`|
|is_related_to   | 0.87  |A mastectomy was performed two months ago."|`mastectomy - two months ago`|


*Example:*

```python
...
re_model = RelationExtractionModel.pretrained("re_oncology_size_wip", "en", "clinical/models") \
    .setInputCols(["embeddings", "pos_tags", "ner_chunk", "dependencies"]) \
    .setOutputCol("relations") \
    .setRelationPairs(["Tumor_Finding-Tumor_Size", "Tumor_Size-Tumor_Finding"]) \
    .setMaxSyntacticDistance(10)
...

sample_text = "The patient presented a 2 cm mass in her left breast, and the tumor in her other breast was 3 cm long."
```

*Results:*

```bash
+----------+-------------+------+-------------+------+----------+
|  relation|      entity1|chunk1|      entity2|chunk2|confidence|
+----------+-------------+------+-------------+------+----------+
|is_size_of|   Tumor_Size|  2 cm|Tumor_Finding|  mass| 0.8532705|
|is_size_of|Tumor_Finding| tumor|   Tumor_Size|  3 cm| 0.8156226|
+----------+-------------+------+-------------+------+----------+
```

</div><div class="h3-box" markdown="1">

####  Brand New `NerQuestionGenerator` Annotator For Automated Prompt Generation For A QA-based Zero-Shot NER Model.

This annotators helps you build questions on the fly using 2 entities from different labels (preferably a subject and a verb). For example, let's suppose you have an NER model, able to detect `PATIENT`and `ADMISSION` in the following text:

`John Smith was admitted Sep 3rd to Mayo Clinic`
- PATIENT: `John Smith`
- ADMISSION: `was admitted`

You can add the following annotator to construct questions using PATIENT and ADMISSION:

```python
# setEntities1 says which entity from NER goes first in the question
# setEntities2 says which entity from NER goes second in the question
# setQuestionMark to True adds a '?' at the end of the sentence (after entity 2)
# To sum up, the pattern is     [QUESTIONPRONOUN] [ENTITY1] [ENTITY2] [QUESTIONMARK]

qagenerator = NerQuestionGenerator()\
  .setInputCols(["ner_chunk"])\
  .setOutputCol("question")\
  .setQuestionMark(True)\
  .setQuestionPronoun("When")\
  .setStrategyType("Paired")\
  .setEntities1(["PATIENT"])\
  .setEntities2(["ADMISSION"])
```
In the column `question` you will find: `When John Smith was admitted?`. Likewise you could have `Where` or any other question pronoun you may need.

You can use those questions in a QuestionAnsweringModel or ZeroShotNER (any model which requires a question as an input. Let's see the case of QA.

```python
qa = BertForQuestionAnswering.pretrained("bert_qa_spanbert_finetuned_squadv1","en") \
  .setInputCols(["question", "document"]) \
  .setOutputCol("answer") \
  .setCaseSensitive(True)
```
The result will be:

```bash
+--------------------------------------------------------+-----------------------------+
|question                                                |answer                       |
+--------------------------------------------------------+-----------------------------+
|[{document, 0, 25, When John Smith was admitted ? ...}] |[{chunk, 0, 8, Sep 3rd ...}] |
+--------------------------------------------------------+-----------------------------+
```
Strategies:
- Paired: First chunk of Entity 1 will be grouped with first chunk of Entity 2, second with second, third with third, etc (one-vs-one)
- Combined: A more flexible strategy to be used in case the number of chukns in Entity 1 is not aligned with the number of chunks in Entityt 2. The first chunk from Entity 1 will be grouped with all chunks in Entity 2, the second chunk in Entity 1 with again be grouped with all the chunks in Entity 2, etc (one-vs-all).

</div><div class="h3-box" markdown="1">

#### Updated ALAB (Annotation Lab) Module Becoming a Fullfledged Suite to Manage Activities on ALAB Via Its API Remotely

We are release a new module for interacting with Annotation Lab with minimal code. Users can now create/edit/delete projects and their tasks. Also, they can upload preannotations, and export annotations and generate training data for various models. Complete documentation and tutorial is available at  [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Annotation_Lab/Complete_ALab_Module_SparkNLP_JSL.ipynb). Following is a comprehensive list of supported tasks:

- Getting details of all projects in the Annotation Lab instance.
- Creating New Projects.
- Deleting Projects.
- Setting & editing configuration of projects.
- Accessing/getting configuration of any existing project.
- Upload tasks to a project.
- Deleting tasks of a project.
- Generating Preannotations for a project using custom Spark NLP pipelines.
- Uploading Preannotations to a project.
- Generating dataset for training Classification models.
- Generating dataset for training NER models.
- Generating dataset for training Assertion models.
- Generating dataset for training Relation Extraction models.

*Using Annotation Lab Module:*

```python
from sparknlp_jsl.alab import AnnotationLab

alab = AnnotationLab()
alab.set_credentials(username=username, password=password, client_secret=client_secret, annotationlab_url=annotationlab_url)

# create a new project
alab.create_project('alab_demo')

# assign ner labels to the project
alab.set_project_config('alab_demo', ner_labels=['Age', 'Gender'])

# upload tasks
alab.upload_tasks('alab_demo', task_list=[txt1, txt2...])

# export tasks
alab.get_annotations('alab_demo')
```

</div><div class="h3-box" markdown="1">

#### New Pretrained Assertion Status Detection Model (`assertion_jsl_augmented`) to Classify The Negativity & Assertion Scope of Medical Concepts

We are releasing new `assertion_jsl_augmented` model to classify the assertion status of the clinical entities with `Present`, `Absent`, `Possible`, `Planned`, `Past`, `Family`, `Hypothetical` and `SomeoneElse` labels.

See [Models Hub Page](https://nlp.johnsnowlabs.com/2022/09/15/assertion_jsl_augmented_en.html) for more details.

*Example:*

```python
...
clinical_assertion = AssertionDLModel.pretrained("assertion_jsl_augmented", "en", "clinical/models")
    .setInputCols(["sentence", "ner_chunk", "embeddings"])
    .setOutputCol("assertion")
...

sample_text = """Patient had a headache for the last 2 weeks, and appears anxious when she walks fast. No alopecia noted.
She denies pain. Her father is paralyzed and it is a stressor for her. She was bullied by her boss and got antidepressant.
We prescribed sleeping pills for her current insomnia"""
```

Results:

```bash
+--------------+-----+---+-------------------------+-----------+---------+
|ner_chunk     |begin|end|ner_label                |sentence_id|assertion|
+--------------+-----+---+-------------------------+-----------+---------+
|headache      |14   |21 |Symptom                  |0          |Past     |
|anxious       |57   |63 |Symptom                  |0          |Possible |
|alopecia      |89   |96 |Disease_Syndrome_Disorder|1          |Absent   |
|pain          |116  |119|Symptom                  |2          |Absent   |
|paralyzed     |136  |144|Symptom                  |3          |Family   |
|antidepressant|212  |225|Drug_Ingredient          |4          |Past     |
|sleeping pills|242  |255|Drug_Ingredient          |5          |Planned  |
|insomnia      |273  |280|Symptom                  |5          |Present  |
+--------------+-----+---+-------------------------+-----------+---------+
```

</div><div class="h3-box" markdown="1">

#### New Chunk Mapper models and Pretrained Pipeline to map entities (phrases) to their corresponding ICD-9, ICD-10-CM and RxNorm codes

We are releasing 4 new chunk mapper models that can map entities to their corresponding ICD-9, ICD-10-CM and RxNorm codes.

|model name|description|
|-|-|
| [rxnorm_normalized_mapper](https://nlp.johnsnowlabs.com/2022/09/29/rxnorm_normalized_mapper_en.html)  |  Mapping drug entities (phrases) with the corresponding **RxNorm codes and normalized resolutions**.  |
| [icd9_mapper](https://nlp.johnsnowlabs.com/2022/09/30/icd9_mapper_en.html)  | Mapping entities with their corresponding ICD-9-CM codes.  |
| [icd10_icd9_mapper](https://nlp.johnsnowlabs.com/2022/09/30/icd10_icd9_mapper_en.html)  | Mapping ICD-10-CM codes with their corresponding ICD-9-CM codes.  |
| [icd9_icd10_mapper](https://nlp.johnsnowlabs.com/2022/09/30/icd9_icd10_mapper_en.html)  | Mapping ICD-9-CM codes with their corresponding ICD-10-CM codes.  |
|  [icd10_icd9_mapping](https://nlp.johnsnowlabs.com/2022/09/30/icd10_icd9_mapping_en.html) (Pipeline) | This pretrained pipeline maps ICD-10-CM codes to ICD-9-CM codes without using any text data. |


*Model Example:*

```python
...
chunkerMapper = ChunkMapperModel.pretrained("rxnorm_normalized_mapper", "en", "clinical/models")\
        .setInputCols(["ner_chunk"])\
        .setOutputCol("mappings")\
        .setRels(["rxnorm_code", "normalized_name"])
...

sample_text = "The patient was given Zyrtec 10 MG, Adapin 10 MG Oral Capsule, Septi-Soothe 0.5 Topical Spray"
```

*Results:*

```bash
+------------------------------+-----------+--------------------------------------------------------------+
|ner_chunk                     |rxnorm_code|normalized_name                                               |
+------------------------------+-----------+--------------------------------------------------------------+
|Zyrtec 10 MG                  |1011483    |cetirizine hydrochloride 10 MG [Zyrtec]                       |
|Adapin 10 MG Oral Capsule     |1000050    |doxepin hydrochloride 10 MG Oral Capsule [Adapin]             |
|Septi-Soothe 0.5 Topical Spray|1000046    |chlorhexidine diacetate 0.5 MG/ML Topical Spray [Septi-Soothe]|
+------------------------------+-----------+--------------------------------------------------------------+
```

*Pipeline Example:*

```python
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline( "icd10_icd9_mapping","en","clinical/models")
pipeline.annotate("Z833 A0100 A000")
```

*Results:*

```bash
| icd10_code          | icd9_code          |
|:--------------------|:-------------------|
| Z833 - A0100 - A000 | V180 - 0020 - 0010 |
```

</div><div class="h3-box" markdown="1">

#### New ICD-9-CM Sentence Entity Resolver Model and Pretrained Pipeline

+ `sbiobertresolve_icd9` : This model maps extracted medical entities to their corresponding ICD-9-CM codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings.

*Example:*

```python
...
icd10_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd9","en", "clinical/models") \
    .setInputCols(["ner_chunk", "sbert_embeddings"]) \
    .setOutputCol("resolution")\
    .setDistanceFunction("EUCLIDEAN")
...

sample_text = "A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus, associated with an acute hepatitis, and obesity with a body mass index (BMI) of 33.5 kg/m2."
```

*Results:*

```bash
+-------------------------------------+-------+---------+------------------------------------------------+----------------------------------------------------------+
|                            ner_chunk| entity|icd9_code|                                      resolution|                                                 all_codes|
+-------------------------------------+-------+---------+------------------------------------------------+----------------------------------------------------------+
|        gestational diabetes mellitus|PROBLEM|   V12.21|[Personal history of gestational diabetes, Ne...|[V12.21, 775.1, 249, 250, 249.7, 249.71, 249.9, 249.61,...|
|subsequent type two diabetes mellitus|PROBLEM|      249|[Secondary diabetes mellitus, Diabetes mellit...|[249, 250, 249.9, 249.7, 775.1, 249.6, 249.8, V12.21, 2...|
|                   an acute hepatitis|PROBLEM|    571.1|[Acute alcoholic hepatitis, Viral hepatitis, ...|[571.1, 070, 571.42, 902.22, 279.51, 571.4, 091.62, 572...|
|                              obesity|PROBLEM|    278.0|[Overweight and obesity, Morbid obesity, Over...|[278.0, 278.01, 278.02, V77.8, 278, 278.00, 272.2, 783....|
|                    a body mass index|PROBLEM|      V85|[Body mass index [BMI], Human bite, Localized...|[V85, E928.3, 278.1, 993, E008.4, V61.5, 747.63, V85.5,...|
+-------------------------------------+-------+---------+------------------------------------------------+----------------------------------------------------------+
```

+ `icd9_resolver_pipeline` : This pretrained pipeline maps entities with their corresponding ICD-9-CM codes. You’ll just feed your text and it will return the corresponding ICD-9-CM codes.

*Example:*

```python
from sparknlp.pretrained import PretrainedPipeline
resolver_pipeline = PretrainedPipeline("icd9_resolver_pipeline", "en", "clinical/models")

sample_text = """A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years and anisakiasis. Also, it was reported that fetal and neonatal hemorrhage"""

result = resolver_pipeline.fullAnnotate(sample_text)
```

Results:

```bash
+-----------------------------+---------+---------+
|chunk                        |ner_chunk|icd9_code|
+-----------------------------+---------+---------+
|gestational diabetes mellitus|PROBLEM  |V12.21   |
|anisakiasis                  |PROBLEM  |127.1    |
|fetal and neonatal hemorrhage|PROBLEM  |772      |
+-----------------------------+---------+---------+
```

</div><div class="h3-box" markdown="1">

#### New Shifting Days Feature in `Deidentification` by Using the New `DocumentHashCoder` Annotator

Now we can shift dates in the documents rather than obfuscating randomly. We have a new `DocumentHashCoder()` annotator to determine shifting days. This annotator gets the hash of the specified column and creates a new document column containing day shift information. And then, the `DeIdentification` annotator deidentifies this new doc. We can use the seed parameter to hash consistently.  

*Example:*

```python
documentHasher = DocumentHashCoder()\
    .setInputCols("document")\
    .setOutputCol("document2")\
    .setPatientIdColumn("patientID")\
    .setRangeDays(100)\
    .setNewDateShift("shift_days")\
    .setSeed(100)

de_identification = DeIdentification() \
    .setInputCols(["ner_chunk", "token", "document2"]) \
    .setOutputCol("deid_text") \
    .setMode("obfuscate") \
    .setObfuscateDate(True) \
    .setDateTag("DATE") \
    .setLanguage("en") \
    .setObfuscateRefSource('faker') \
    .setUseShifDays(True)
```

*Results:*

```bash

output.select('patientID','text', 'deid_text.result').show(truncate = False)
+---------+----------------------------------------+---------------------------------------------+
|patientID|text                                    |result                                       |
+---------+----------------------------------------+---------------------------------------------+
|A001     |Chris Brown was discharged on 10/02/2022|[Glorious Mc was discharged on 27/03/2022]   |
|A001     |Mark White was discharged on 10/04/2022 |[Kimberlee Bair was discharged on 25/05/2022]|
|A003     |John was discharged on 15/03/2022       |[Monia Richmond was discharged on 17/05/2022]|
|A003     |John Moore was discharged on 15/12/2022 |[Veleta Pollard was discharged on 16/02/2023]|
+---------+----------------------------------------+---------------------------------------------+
```

Instead of shifting days according to ID column, we can specify shifting values with another column.

*Example:*

```python
documentHasher = DocumentHashCoder()\
    .setInputCols("document")\
    .setOutputCol("document2")\
    .setDateShiftColumn("dateshift")\

de_identification = DeIdentification() \
    .setInputCols(["ner_chunk", "token", "document2"]) \
    .setOutputCol("deid_text") \
    .setMode("obfuscate") \
    .setObfuscateDate(True) \
    .setDateTag("DATE") \
    .setLanguage("en") \
    .setObfuscateRefSource('faker') \
    .setUseShifDays(True)
```

*Results:*

```bash
+----------------------------------------+---------+---------------------------------------------+
|text                                    |dateshift|result                                       |
+----------------------------------------+---------+---------------------------------------------+
|Chris Brown was discharged on 10/02/2022|10       |[Levorn Powers was discharged on 20/02/2022] |
|Mark White was discharged on 10/04/2022 |10       |[Hall Jointer was discharged on 20/04/2022]  |
|John was discharged on 15/03/2022       |30       |[Jared Gains was discharged on 14/04/2022]   |
|John Moore was discharged on 15/12/2022 |30       |[Frederic Seitz was discharged on 14/01/2023]|
+----------------------------------------+---------+---------------------------------------------+
```

You can check [Clinical Deidentification Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb) for more examples.

</div><div class="h3-box" markdown="1">

#### Updated NER Model Finder Pretrained Pipeline to Help Users Find The Most Appropriate NER Model For Their Use Case In One-Liner

We have updated `ner_model_finder` pretrained pipeline and `sbertresolve_ner_model_finder` resolver model with 70 clinical NER models and their labels.

See [Models Hub Page](https://nlp.johnsnowlabs.com/2022/09/05/ner_model_finder_en.html) for more details and the [Pretrained Clinical Pipelines Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb) for the examples.

</div><div class="h3-box" markdown="1">

#### Support Different Version and Year Combinations on Medicare Risk Adjustment Score Calculation Module

Now, you can calculate CMS-HCC risk score with different version and year combinations by importing one of the following function calculate the score.

```
- profileV2217   - profileV2318  - profileV2417
- profileV2218   - profileV2319  - profileV2418
- profileV2219                   - profileV2419
- profileV2220                   - profileV2420
- profileV2221                   - profileV2421
- profileV2222                   - profileV2422
```

```python
from sparknlp_jsl.functions import profileV24Y20
```
See the [notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.1.Calculate_Medicare_Risk_Adjustment_Score.ipynb) for more details.

</div><div class="h3-box" markdown="1">

#### Core Improvements and Bug Fixes

+ `ContextualParserApproach`:  New parameter `completeContextMatch`.  
This parameter let the user define whether to do an **exact match of prefix and suffix**.

+ `Deidentification`: Enhanced default regex rules in French deidentification for `DATE` entity extraction.

+ `ZeroShotRelationExtractionModel`: Fixed the issue that setting some parameters together and no need to `setRelationalCategories` after downloading the model.  

</div><div class="h3-box" markdown="1">

#### New and Updated Notebooks

+ New [MedicalBertForSequenceClassification Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/19.MedicalBertForSequenceClassification_in_SparkNLP.ipynb) to show how to use `MedicalBertForSequenceClassification` models.
+ New [ALAB Module Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Annotation_Lab/Complete_ALab_Module_SparkNLP_JSL.ipynb) to show all features of ALAB Module.
+ Updated [Medicare Risk Adjustment Score Calculation Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.1.Calculate_Medicare_Risk_Adjustment_Score.ipynb) with the new changes in HCC score calculation functions.
+ Updated [Clinical DeIdentification Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb) by adding how not to deidentify a part of an entity section and showing examples of shifting days feature with the new `DocumentHashCoder`.  
+ Updated [Pretrained Clinical Pipelines Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb) with the updated `ner_model_finder` results.

</div><div class="h3-box" markdown="1">

#### 50+ New Clinical Models and Pipelines Added & Updated in Total

+ `assertion_jsl_augmented`
+ `rxnorm_normalized_mapper`
+ `ner_model_finder`
+ `sbertresolve_ner_model_finder`
+ `sbiobertresolve_icd9`
+ `icd9_resolver_pipeline`
+ `rxnorm_normalized_mapper`
+ `icd9_mapper`
+ `icd10_icd9_mapper`
+ `icd9_icd10_mapper`
+ `icd10_icd9_mapping`
+ `bert_qa_spanbert_finetuned_squadv1`
+ `ner_oncology_therapy_wip`
+ `ner_oncology_diagnosis_wip`
+ `ner_oncology_wip`
+ `ner_oncology_tnm_wip`
+ `ner_oncology_anatomy_general_wip`
+ `ner_oncology_demographics_wip`
+ `ner_oncology_test_wip`
+ `ner_oncology_unspecific_posology_wip`
+ `ner_oncology_anatomy_granular_wip`
+ `ner_oncology_response_to_treatment_wip`
+ `ner_oncology_biomarker_wip`
+ `ner_oncology_posology_wip`
+ `bert_token_classifier_ner_oncology_therapy_wip`
+ `bert_token_classifier_ner_oncology_diagnosis_wip`
+ `bert_token_classifier_ner_oncology_wip`
+ `bert_token_classifier_ner_oncology_tnm_wip`
+ `bert_token_classifier_ner_oncology_anatomy_general_wip`
+ `bert_token_classifier_ner_oncology_demographics_wip`
+ `bert_token_classifier_ner_oncology_test_wip`
+ `bert_token_classifier_ner_oncology_unspecific_posology_wip`
+ `bert_token_classifier_ner_oncology_anatomy_granular_wip`
+ `bert_token_classifier_ner_oncology_response_to_treatment_wip`
+ `bert_token_classifier_ner_oncology_biomarker_wip`
+ `bert_token_classifier_ner_oncology_posology_wip`
+ `assertion_oncology_wip`
+ `assertion_oncology_problem_wip`
+ `assertion_oncology_treatment_wip`
+ `assertion_oncology_response_to_treatment_wip`
+ `assertion_oncology_test_binary_wip`
+ `assertion_oncology_smoking_status_wip`
+ `assertion_oncology_family_history_wip`
+ `assertion_oncology_demographic_binary_wip`
+ `re_oncology_size_wip`
+ `re_oncology_biomarker_result_wip`
+ `re_oncology_granular_wip`
+ `re_oncology_location_wip`
+ `re_oncology_temporal_wip`
+ `re_oncology_test_result_wip`
+ `re_oncology_wip`
+ `redl_oncology_size_biobert_wip`
+ `redl_oncology_biomarker_result_biobert_wip`
+ `redl_oncology_location_biobert_wip`
+ `redl_oncology_temporal_biobert_wip`
+ `redl_oncology_test_result_biobert_wip`
+ `redl_oncology_biobert_wip`
+ `redl_oncology_granular_biobert_wip`

</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_4_1_0">Version 4.1.0</a>
    </li>
    <li>
        <strong>Version 4.2.0</strong>
    </li>
</ul>
<ul class="pagination owl-carousel pagination_big">
    <li class="active"><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li><a href="release_notes_3_5_1">3.5.1</a></li>
    <li><a href="release_notes_3_5_0">3.5.0</a></li>
    <li><a href="release_notes_3_4_2">3.4.2</a></li>
    <li><a href="release_notes_3_4_1">3.4.1</a></li>
    <li><a href="release_notes_3_4_0">3.4.0</a></li>
    <li><a href="release_notes_3_3_4">3.3.4</a></li>
    <li><a href="release_notes_3_3_2">3.3.2</a></li>
    <li><a href="release_notes_3_3_1">3.3.1</a></li>
    <li><a href="release_notes_3_3_0">3.3.0</a></li>
    <li><a href="release_notes_3_2_3">3.2.3</a></li>
    <li><a href="release_notes_3_2_2">3.2.2</a></li>
    <li><a href="release_notes_3_2_1">3.2.1</a></li>
    <li><a href="release_notes_3_2_0">3.2.0</a></li>
    <li><a href="release_notes_3_1_3">3.1.3</a></li>
    <li><a href="release_notes_3_1_2">3.1.2</a></li>
    <li><a href="release_notes_3_1_1">3.1.1</a></li>
    <li><a href="release_notes_3_1_0">3.1.0</a></li>
    <li><a href="release_notes_3_0_3">3.0.3</a></li>
    <li><a href="release_notes_3_0_2">3.0.2</a></li>
    <li><a href="release_notes_3_0_1">3.0.1</a></li>
    <li><a href="release_notes_3_0_0">3.0.0</a></li>
    <li><a href="release_notes_2_7_6">2.7.6</a></li>
    <li><a href="release_notes_2_7_5">2.7.5</a></li>
    <li><a href="release_notes_2_7_4">2.7.4</a></li>
    <li><a href="release_notes_2_7_3">2.7.3</a></li>
    <li><a href="release_notes_2_7_2">2.7.2</a></li>
    <li><a href="release_notes_2_7_1">2.7.1</a></li>
    <li><a href="release_notes_2_7_0">2.7.0</a></li>
    <li><a href="release_notes_2_6_2">2.6.2</a></li>
    <li><a href="release_notes_2_6_0">2.6.0</a></li>
    <li><a href="release_notes_2_5_5">2.5.5</a></li>
    <li><a href="release_notes_2_5_3">2.5.3</a></li>
    <li><a href="release_notes_2_5_2">2.5.2</a></li>
    <li><a href="release_notes_2_5_0">2.5.0</a></li>
    <li><a href="release_notes_2_4_6">2.4.6</a></li>
    <li><a href="release_notes_2_4_5">2.4.5</a></li>
    <li><a href="release_notes_2_4_2">2.4.2</a></li>
    <li><a href="release_notes_2_4_1">2.4.1</a></li>
    <li><a href="release_notes_2_4_0">2.4.0</a></li>
</ul>