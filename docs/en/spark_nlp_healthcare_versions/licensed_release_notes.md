---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes
permalink: /docs/en/spark_nlp_healthcare_versions/licensed_release_notes
key: docs-licensed-release-notes
modify_date: 2023-02-13
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## 4.3.0

#### Highlights

+ 12 new clinical models and pipelines added & updated (8 new clinical named entity recognition models including 4 social determinants of health models)
+ New Chunk Mapper model for mapping RxNorm codes to drug brand names
+ New text classification annotators (architectures) for training text classification models using SVM and Logistic Regression with sentence embeddings
+ One-liner clinical deidentification module
+ Certification_Training notebooks (written in johnsnowlabs library) moved to parent workshop folder
+ Different validation split per epoch in `MedicalNerApproach`
+ Core improvements and bug fixes
    - New read_conll method for reading conll files as `Conll.readDataset` does but it returns pandas dataframe with document(task) ids.
    - Updated documentation
    - Allow using `FeatureAssembler` in pretrained pipelines.
    - Fixed `RelationExtractionModel` running in LightPipeline
    - Fixed `get_conll_data` method issue
+ New and updated notebooks
    - New [Clinical Deidentification Utility Module Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.5.Clinical_Deidentification_Utility_Module.ipynb).
    - Updated [Clinical_Named_Entity_Recognition_Model](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb) with `Conll.readDataset` examples.
    - Updated [Clinical Text Classification with Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/30.Clinical_Text_Classification_with_Spark_NLP.ipynb) with new `GenericLogRegClassifierApproach` and `GenericSVMClassifierApproach` examples.
+ New and updated demos
    - [SOCIAL DETERMINANT NER](https://demo.johnsnowlabs.com/healthcare/SOCIAL_DETERMINANT_NER/) demo
    - [SOCIAL DETERMINANT CLASSIFICATION](https://demo.johnsnowlabs.com/healthcare/SOCIAL_DETERMINANT_CLASSIFICATION/) demo
    - [SOCIAL DETERMINANT GENERIC CLASSIFICATION](https://demo.johnsnowlabs.com/healthcare/SOCIAL_DETERMINANT_GENERIC_CLASSIFICATION/) demo
+ 13 new clinical models and pipelines added & updated in total


#### 12 New Clinical Models And Pipelines Added & Updated (8 New Clinical Named Entity Recognition Models Including 4 Social Determinants of Health Models)


+ We are releasing 4 new SDOH NER models that were trained by using `embeddings_clinical` embeddings model.


| model name                                     | description                                                                                         | predicted entities                     |
|------------------------------------------------|-----------------------------------------------------------------------------------------------------|----------------------------------------|
| [ner_sdoh_wip](https://nlp.johnsnowlabs.com/2023/02/11/ner_sdoh_wip_en.html) | Extracts terminology related to Social Determinants of Health from various kinds of biomedical documents. | `Other_SDoH_Keywords` `Education` `Population_Group` `Quality_Of_Life` `Housing` `Substance_Frequency` `Smoking` `Eating_Disorder` `Obesity` `Healthcare_Institution` `Financial_Status` `Age` `Chidhood_Event` `Exercise` `Communicable_Disease` `Hypertension` `Other_Disease` `Violence_Or_Abuse` `Spiritual_Beliefs` `Employment` `Social_Exclusion` `Access_To_Care` `Marital_Status` `Diet` `Social_Support` `Disability` `Mental_Health` `Alcohol` `Insurance_Status` `Substance_Quantity` `Hyperlipidemia` `Family_Member` `Legal_Issues` `Race_Ethnicity` `Gender` `Geographic_Entity` `Sexual_Orientation` `Transportation` `Sexual_Activity` `Language` `Substance_Use`|
| [ner_sdoh_social_environment_wip](https://nlp.johnsnowlabs.com/2023/02/10/ner_sdoh_social_environment_wip_en.html)     | Extracts social environment terminologies related to Social Determinants of Health from various kinds of biomedical documents.     | `Social_Support` `Chidhood_Event` `Social_Exclusion` `Violence_Abuse_Legal`        |
| [ner_sdoh_demographics_wip](https://nlp.johnsnowlabs.com/2023/02/10/ner_sdoh_demographics_wip_en.html)                 | Extracts demographic information related to Social Determinants of Health from various kinds of biomedical documents.              | `Family_Member` `Age` `Gender` `Geographic_Entity` `Race_Ethnicity`  `Language` `Spiritual_Beliefs`   |
| [ner_sdoh_income_social_status_wip](https://nlp.johnsnowlabs.com/2023/02/10/ner_sdoh_income_social_status_wip_en.html) | Extracts income and social status information related to Social Determinants of Health from various kinds of biomedical documents. | `Education` `Marital_Status` `Financial_Status` `Population_Group` `Employment` |


*Example*:

```python
...
clinical_embeddings = WordEmbeddingsModel.pretrained("embeddings_clinical", "en", "clinical/models")\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("embeddings")

ner_model = MedicalNerModel.pretrained("ner_sdoh_wip", "en", "clinical/models")\
    .setInputCols(["sentence", "token","embeddings"])\
    .setOutputCol("ner")

sample_texts ="Smith is a 55 years old, divorced Mexcian American woman with financial problems. She speaks spanish. She lives in an apartment. She has been struggling with diabetes for the past 10 years and has recently been experiencing frequent hospitalizations due to uncontrolled blood sugar levels. Smith works as a cleaning assistant and does not have access to health insurance or paid sick leave. She has a son student at college. Pt with likely long-standing depression. She is aware she needs rehab. Pt reprots having her catholic faith as a means of support as well.  She has long history of etoh abuse, beginning in her teens. She reports she has been a daily drinker for 30 years, most recently drinking beer daily. She smokes a pack of cigarettes a day. She had DUI back in April and was due to be in court this week."
```

*Result*:

```bash
+------------------+-----+---+-------------------+
|chunk             |begin|end|ner_label          |
+------------------+-----+---+-------------------+
|55 years old      |11   |22 |Age                |
|divorced          |25   |32 |Marital_Status     |
|Mexcian American  |34   |49 |Race_Ethnicity     |
|financial problems|62   |79 |Financial_Status   |
|spanish           |93   |99 |Language           |
|apartment         |118  |126|Housing            |
|diabetes          |158  |165|Other_Disease      |
|cleaning assistant|307  |324|Employment         |
|health insurance  |354  |369|Insurance_Status   |
|son               |401  |403|Family_Member      |
|student           |405  |411|Education          |
|college           |416  |422|Education          |
|depression        |454  |463|Mental_Health      |
|rehab             |489  |493|Access_To_Care     |
|catholic faith    |518  |531|Spiritual_Beliefs  |
|support           |547  |553|Social_Support     |
|etoh abuse        |589  |598|Alcohol            |
|teens             |618  |622|Age                |
|drinker           |658  |664|Alcohol            |
|drinking beer     |694  |706|Alcohol            |
|daily             |708  |712|Substance_Frequency|
|smokes            |719  |724|Smoking            |
|a pack            |726  |731|Substance_Quantity |
|cigarettes        |736  |745|Smoking            |
|a day             |747  |751|Substance_Frequency|
|DUI               |762  |764|Legal_Issues       |
+------------------+-----+---+-------------------+
```

+ We are releasing 8 new NER models which are trained by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.

    - ner_eu_clinical_case: This model extracts 6 different clinical entities based on medical taxonomies.

    - ner_eu_clinical_condition: This model extracts one entity – clinical / medical conditions.

| model name                    | lang | predicted entities    |
|------------------------------ |------|-----------------------|
| [ner_eu_clinical_case](https://nlp.johnsnowlabs.com/2023/02/01/ner_eu_clinical_case_es.html)      | es   | `clinical_condition` `clinical_event` `bodypart` `units_measurements` `patient` `date_time` |
| [ner_eu_clinical_case](https://nlp.johnsnowlabs.com/2023/02/01/ner_eu_clinical_case_fr.html)      | fr   | `clinical_condition` `clinical_event` `bodypart` `units_measurements` `patient` `date_time` |
| [ner_eu_clinical_case](https://nlp.johnsnowlabs.com/2023/02/02/ner_eu_clinical_case_eu.html)      | eu   | `clinical_condition` `clinical_event` `bodypart` `units_measurements` `patient` `date_time` |
| [ner_eu_clinical_condition](https://nlp.johnsnowlabs.com/2023/02/06/ner_eu_clinical_condition_en.html) | en   | `clinical_condition`  |
| [ner_eu_clinical_condition](https://nlp.johnsnowlabs.com/2023/02/06/ner_eu_clinical_condition_es.html) | es   | `clinical_condition`  |
| [ner_eu_clinical_condition](https://nlp.johnsnowlabs.com/2023/02/06/ner_eu_clinical_condition_eu.html) | eu   | `clinical_condition`  |
| [ner_eu_clinical_condition](https://nlp.johnsnowlabs.com/2023/02/06/ner_eu_clinical_condition_fr.html) | fr   | `clinical_condition`  |
| [ner_eu_clinical_condition](https://nlp.johnsnowlabs.com/2023/02/06/ner_eu_clinical_condition_it.html) | it   | `clinical_condition`  |


*Example*:

```python
word_embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d","es")\
	.setInputCols(["sentence","token"])\
	.setOutputCol("embeddings")

ner = MedicalNerModel.pretrained("ner_eu_clinical_case", "es", "clinical/models") \
	.setInputCols(["sentence", "token", "embeddings"]) \
	.setOutputCol("ner")

sample_text = """Paciente de 59 años que refiere dificultad para caminar desde hace un mes aproximadamente. Presenta debilidad y dolor en los miembros inferiores, que mejora tras detenerse, acompañándose en ocasiones de lumbalgia no irradiada. En la exploración neurológica presenta habla hipofónica, facial centrado. Debido a la mala perfusión secundaria a la sepsis aparecieron lesiones necróticas en extremidades superiores y principalmente inferiores distales. Motilidad ocular interna y externa normal."""
```

*Result*:

```bash
+---------------------------+------------------+
|chunk                      |ner_label         |
+---------------------------+------------------+
|Paciente de 59 años        |patient           |
|refiere                    |clinical_event    |
|dificultad para caminar    |clinical_event    |
|hace un mes aproximadamente|date_time         |
|debilidad                  |clinical_event    |
|dolor                      |clinical_event    |
|los miembros inferiores    |bodypart          |
|mejora                     |clinical_event    |
|detenerse                  |clinical_event    |
|lumbalgia                  |clinical_event    |
|irradiada                  |clinical_event    |
|exploración                |clinical_event    |
|habla                      |clinical_event    |
|hipofónica                 |clinical_event    |
|perfusión                  |clinical_event    |
|sepsis                     |clinical_event    |
|lesiones                   |clinical_event    |
|extremidades superiores    |bodypart          |
|inferiores distales        |bodypart          |
|Motilidad                  |clinical_event    |
|normal                     |units_measurements|
+---------------------------+------------------+

```



#### New Chunk Mapper Model for Mapping RxNorm Codes to Drug Brand Names

We are releasing `rxnorm_drug_brandname_mapper` pretrained model that maps RxNorm and RxNorm Extension codes with their corresponding drug brand names. <br/>
It returns 2 types of brand names called `rxnorm_brandname` and `rxnorm_extension_brandname` for the corresponding RxNorm or RxNorm Extension code.

*Example*:

```python
...

chunkerMapper = ChunkMapperModel.pretrained("rxnorm_drug_brandname_mapper", "en", "clinical/models")\
       .setInputCols(["rxnorm_chunk"])\
       .setOutputCol("mappings")\
       .setRels(["rxnorm_brandname", "rxnorm_extension_brandname"])

sample_text= ['metformin', 'advil']
```

*Result:*

```bash
+--------------+-------------+--------------------------------------------------+--------------------------+
|     drug_name|rxnorm_result|                                    mapping_result|                 relation |
+--------------+-------------+--------------------------------------------------+--------------------------+
|     metformin|         6809|Actoplus Met (metformin):::Avandamet (metformin...|          rxnorm_brandname|
|     metformin|         6809|A FORMIN (metformin):::ABERIN MAX (metformin)::...|rxnorm_extension_brandname|
|         advil|       153010|                                     Advil (Advil)|          rxnorm_brandname|
|         advil|       153010|                                              NONE|rxnorm_extension_brandname|
+--------------+-------------+--------------------------------------------------+--------------------------+
 ```



#### New Text Classification Annotators (Architectures) For Training Text Classification Models Using SVM and Logistic Regression With Sentence Embeddings

+ We have a new text classification architecture called `GenericLogRegClassifierApproach` that implements a multinomial *Logistic Regression* with sentence embeddings. This is a single layer neural network with the logistic function at the output. The input to the model is `FeatureVector` (from any sentence embeddings) and the output is `Category` annotations with labels and corresponding confidence scores. Training data requires "text" and their "label" columns only and the trained model will be a `GenericLogRegClassifierModel()`.


+ We have another text classification architecture called `GenericSVMClassifierApproach` that implements *SVM (Support Vector Machine)* classification. The input to the model is `FeatureVector` (from any sentence embeddings) and the output is `Category` annotations with labels and corresponding confidence scores. Taining data requires "text" and their "label" columns only and the trained model will be a `GenericSVMClassifierModel()`.

Input types: `FEATURE_VECTOR`
Output type: `CATEGORY`


*Example*:

```python

features_asm =  sparknlp_jsl.base.FeaturesAssembler()\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("feature_vector")

gcf_graph_builder = sparknlp_jsl.annotators.TFGraphBuilder()\
    .setModelName("logreg_classifier")\
    .setInputCols(["feature_vector"]) \
    .setLabelColumn("label")\
    .setGraphFolder("/tmp/")\
    .setGraphFile("log_reg_graph.pb")\

log_reg_approach = sparknlp_jsl.annotators.GenericLogRegClassifierApproach()\
    .setLabelColumn("label")\
    .setInputCols(["feature_vector"])\
    .setOutputCol("prediction")\
    .setModelFile(f"/tmp/log_reg_graph.pb")\
    .setEpochsNumber(10)\
    .setBatchSize(1)\
    .setLearningRate(0.001)

```


#### One-Liner Clinical Deidentification Module

Spark NLP for Healthcare provides functionality to apply Deidentification using one-liner module called `Deid`. <br/>

The `Deid` module is a tool for deidentifying Protected Health Information (PHI) from data in a file path. It can be used with or without ant Spark NLP NER pipelines. It can apply deidentification and obfuscation on different columns at the same time.
It returns the deidentification & obfuscation results as a spark dataframe as well as a `csv` or `json file` saved locally.
The module also includes functionality for applying Structured Deidentification task to data from a file path. <br/>

The function, `deidentify()`, can be used with a custom pipeline or without defining any custom pipeline. <br/>
`structured_deidentifier()` function can be used for the Structured Deidentification task. <br/>


Please see [this notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.5.Clinical_Deidentification_Utility_Module.ipynb) for the detailed usage and explanation of all parameters. <br/>
Check here for the [documentation](https://nlp.johnsnowlabs.com/docs/en/utility_helper_modules) of the module.

- Deidentification with a custom pipeline

*Example*:

```python
from sparknlp_jsl import Deid

deid_implementor= Deid(
# required: Spark session with spark-nlp-jsl jar
spark
)

res= deid_implementor.deidentify(
# required: The path of the input file. Default is None. File type must be 'csv' or 'json'.
input_file_path="data.csv",

#optional:  The path of the output file. Default is 'deidentified.csv'. File type must be 'csv' or 'json'.
output_file_path="deidentified.csv",

#optional: The separator of the input csv file. Default is "\t".
separator=",",

#optional: A custom pipeline model to be used for deidentification. If not specified, the default is None.
custom_pipeline=nlpModel,

#optional: Fields to be deidentified and their deidentification modes, by default {"text": "mask"}
fields={"text_column_1": "text_column_1_deidentified", "text_column_2": "text_column_2_deidentified"},

#optional:  The masking policy. Default is "entity_labels".
masking_policy="fixed_length_chars",

#optional: The fixed mask length. Default is 4.
fixed_mask_length=4)

```

*Result:*

```bash
+---+----------------------------------------------------------------------+----------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
| ID|                                                         text_column_1|                    text_column_1_deidentified|                                                         text_column_2|                                            text_column_2_deidentified|
+---+----------------------------------------------------------------------+----------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+
|  0|Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson ...|Record date : ** , ** , M.D . , Name : ** MR .|Date : 01/13/93 PCP : Oliveira , 25 years-old , Record date : 2079-...|Date : 10-16-1991 PCP : Alveda Castles , 26 years-old , Record date...|
+---+----------------------------------------------------------------------+----------------------------------------------+----------------------------------------------------------------------+----------------------------------------------------------------------+

```

- Deidentification with no custom pipeline

*Example*:


```python
from sparknlp_jsl import Deid

deid_implementor= Deid(
# required: Spark session with spark-nlp-jsl jar
spark
)

res= deid_implementor.deidentify(
# required: The path of the input file. Default is None. File type must be 'csv' or 'json'.
input_file_path="data.csv",

#optional:  The path of the output file. Default is 'deidentified.csv'. File type must be 'csv' or 'json'.
output_file_path="deidentified.csv",

#optional: The separator of the input csv file. Default is "\t".
separator=",",

#optional: Fields to be deidentified and their deidentification modes, by default {"text": "mask"}
fields={"text": "mask"},

#optional: The masking policy. Default is "entity_labels".
masking_policy="entity_labels")

```

*Result:*

```
+---+----------------------------------------------------------------------+----------------------------------------------------------------------+
| ID|                                                         text_original|                                                             text_deid|
+---+----------------------------------------------------------------------+----------------------------------------------------------------------+
|  0|                                                                     "|                                                                     "|
|  1|Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson ...|Record date : <DATE> , <DOCTOR> , M.D . , Name : <PATIENT> , MR # <...|
|  2|                                                                     "|                                                                     "|
+---+----------------------------------------------------------------------+----------------------------------------------------------------------+

```


- Structured Deidentification

*Example*:

```python
from sparknlp_jsl import Deid

deid_implementor= Deid(
# required: Spark session with spark-nlp-jsl jar
spark
)

res= deid_implementor.structured_deidentifier(

#required: The path of the input file. Default is None. File type must be 'csv' or 'json'.
input_file_path="data.csv",

#optional:  The path of the output file. Default is 'deidentified.csv'. File type must be 'csv' or 'json'.
output_file_path="deidentified.csv",

#optional: The separator of the input csv file. Default is "\t".
separator=",",

#optional: A dictionary that contains the column names and the tags that should be used for deidentification. Default is {"NAME":"PATIENT","AGE":"AGE"}
columns_dict= {"NAME": "ID", "DOB": "DATE"},

#optional: The seed value for the random number generator. Default is {"NAME": 23, "AGE": 23}
columns_seed= {"NAME": 23, "DOB": 23},

#optional: The source of the reference file. Default is faker.
ref_source="faker",

#optional: The number of days to be shifted. Default is None
shift_days=5)

```

*Result:*

```bash
+----------+------------+--------------------+---+----------------+
|      NAME|         DOB|             ADDRESS|SBP|             TEL|
+----------+------------+--------------------+---+----------------+
|[N2649912]|[18/02/1977]|       711 Nulla St.|140|      673 431234|
| [W466004]|[28/02/1977]|     1 Green Avenue.|140|+23 (673) 431234|
| [M403810]|[16/04/1900]|Calle del Liberta...|100|      912 345623|
+----------+------------+--------------------+---+----------------+
```

#### Different Validation Split Per Epoch In `MedicalNerApproach`

The validation splits in `MedicalNerApproach` used to be static and same for every epoch. Now we can control with behaviour with a new parameter called `setRandomValidationSplitPerEpoch(bool)` and allow users  to set random validation splits per epoch.

#### Certification_Training Notebooks (Written In Johnsnowlabs Library) Moved to Parent Workshop Folder

- re-organize and re-locate [open-source-nlp](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/open-source-nlp) folder
- re-organize and re-locate [healthcare-nlp](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/healthcare-nlp) folder




#### Core Improvements and Bug Fixes

- New read_conll method for reading conll files as `Conll.readDataset` does but it returns dataframe with document(task) ids.
- Updated documentation
- Allow using `FeatureAssembler` in pretrained pipelines.
- Fixed `RelationExtractionModel` running in LightPipeline
- Fixed `get_conll_data` method issue


#### New and Updated Notebooks

- New [Clinical Deidentification Utility Module Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.5.Clinical_Deidentification_Utility_Module.ipynb).
- Updated [Clinical_Named_Entity_Recognition_Model](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb) with `Conll.readDataset` examples.
- Updated [Clinical Text Classification with Spark NLP](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/30.Clinical_Text_Classification_with_Spark_NLP.ipynb) with new `GenericLogRegClassifierApproach` and `GenericSVMClassifierApproach` examples.


#### New and Updated Demos

+ [SOCIAL DETERMINANT NER](https://demo.johnsnowlabs.com/healthcare/SOCIAL_DETERMINANT_NER/) demo
+ [SOCIAL DETERMINANT CLASSIFICATION](https://demo.johnsnowlabs.com/healthcare/SOCIAL_DETERMINANT_CLASSIFICATION/) demo
+ [SOCIAL DETERMINANT GENERIC CLASSIFICATION](https://demo.johnsnowlabs.com/healthcare/SOCIAL_DETERMINANT_GENERIC_CLASSIFICATION/) demo


#### 12 New Clinical Models and Pipelines Added & Updated in Total

+ `ner_eu_clinical_case`-> es
+ `ner_eu_clinical_case`-> fr
+ `ner_eu_clinical_case`-> eu
+ `ner_eu_clinical_condition`-> en
+ `ner_eu_clinical_condition`-> es
+ `ner_eu_clinical_condition`-> fr
+ `ner_eu_clinical_condition`-> eu
+ `ner_eu_clinical_condition`-> it
+ `ner_sdoh_demographics_wip`
+ `ner_sdoh_income_social_status_wip`
+ `ner_sdoh_social_environment_wip`
+ `ner_sdoh_wip`
+ `rxnorm_drug_brandname_mapper`




For all Spark NLP for Healthcare models, please check: [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Healthcare+NLP)



</div>
<div class="prev_ver h3-box" markdown="1">

## Previous versions

</div>
{%- include docs-healthcare-pagination.html -%}