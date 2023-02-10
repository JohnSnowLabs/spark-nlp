---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 4.2.8
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_4_2_8
key: docs-licensed-release-notes
modify_date: 2023-01-26
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## 4.2.8

#### Highlights

+ 4 new clinical named entity recognition models (3 oncology, 1 others)
+ 5 new Social Determenant of Health text classification models
+ New `DocumentMLClassifierApproach` annotator for training text classification models using SVM and Logistic Regression using TfIdf
+ New `Resolution2Chunk` annotator to map entity resolver outputs (terminology codes) to other clinical terminologies
+ New `DocMapperModel` annotator allows to use any mapper model in `DOCUMENT` type
+ Option to return `Deidentification` output as a single document 
+ Inter-Annotator Agreement (IAA) metrics module that works with NLP Lab seamlessly
+ Assertion dataset preparation module now supports chunk start and end indices, rather than token indices
+ Added `ner_source` in the `ChunkConverter` metadata
+ Core improvements and bug fixes
    - Added chunk confidence score in the `RelationExtractionModel` metadata
    - Added confidence score in the `DocumentLogRegClassifierApproach` metadata
    - Fixed non-deterministic Relation Extraction DL Models (30+ models updated in the model hub)
    - Fixed incompatible PretrainedPipelines with PySpark v3.2.x and v3.3.x 
    - Fixed `ZIP` label issue on `faker` mode with `setZipCodeTag` parameter in `Deidentification` 
    - Fixed obfuscated numbers have the same number of chars as the original ones 
    - Fixed name obfuscation hashes in `Deidentification` for romanian language 
    - Fixed LightPipeline validation parameter for internal annotators
    - LightPipeline support for `GenericClassifier` (`FeatureAssembler`) 
+ New and updated notebooks
    -  New [Clinical Text Classification with Spark_NLP Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/30.Clinical_Text_Classification_with_Spark_NLP.ipynb) 
    - New [Clinical Text Classification with DocumentMLClassifier Notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/30.1.Text_Classification_with_DocumentMLClassifier.ipynb)
    - Updated [ALAB Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Annotation_Lab/Complete_ALab_Module_SparkNLP_JSL.ipynb) 
+ New and updated demos
    - [SOCIAL DETERMINANT](https://demo.johnsnowlabs.com/healthcare/SOCIAL_DETERMINANT/) demo
+ 9 new clinical models and pipelines added & updated in total


#### 4 New Clinical Named Entity Recognition Models (3 Oncology, 1 Others)

- We are releasing 3 new oncological NER models that were trained by using `embeddings_healthcare_100d` embeddings model.

| model name                                     | description                                                                                         | predicted entities                     |
|----------------------------------------------- |-----------------------------------------------------------------------------------------------------|--------------------------------------- |
| [ner_oncology_anatomy_general_healthcare](https://nlp.johnsnowlabs.com/2023/01/11/ner_oncology_anatomy_general_healthcare_en.html)    | Extracts anatomical entities using an unspecific label                                              | `Anatomical_Site` `Direction`          |
| [ner_oncology_biomarker_healthcare](https://nlp.johnsnowlabs.com/2023/01/11/ner_oncology_biomarker_healthcare_en.html)          | Extracts mentions of biomarkers and biomarker results in oncological texts.                         | `Biomarker_Result` `Biomarker`         |
| [ner_oncology_unspecific_posology_healthcare](https://nlp.johnsnowlabs.com/2023/01/11/ner_oncology_unspecific_posology_healthcare_en.html)| Extracts mentions of treatments and posology information using unspecific labels (low granularity). | `Posology_Information` `Cancer_Therapy`|

*Example*:

```python
...
word_embeddings = WordEmbeddingsModel()\
    .pretrained("embeddings_healthcare_100d", "en", "clinical/models")\
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("embeddings")  

ner = MedicalNerModel\
    .pretrained("ner_oncology_anatomy_general_healthcare", "en", "clinical/models") \
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")

text = "The patient presented a mass in her left breast, and a possible metastasis in her lungs and in her liver."
```

*Result*:

```bash
+------------------+----------------+
|chunk             |ner_label       |
+------------------+----------------+
|left              |Direction       |
|breast            |Anatomical_Site |
|lungs             |Anatomical_Site |
|liver             |Anatomical_Site |
+------------------+----------------+
```


- We are releasing new oncological NER models that used for model training is provided by European Clinical Case Corpus (E3C), a project aimed at offering a freely available multilingual corpus of semantically annotated clinical narratives.


*Example*:

```python
...
ner = MedicalNerModel.pretrained('ner_eu_clinical_case', "en", "clinical/models") \
	.setInputCols(["sentence", "token", "embeddings"]) \
	.setOutputCol("ner")

text = """A 3-year-old boy with autistic disorder on hospital of pediatric ward A at university hospital. He has no family history of illness or autistic spectrum disorder."""
```

*Result*:

```bash
+------------------------------+------------------+
|chunk                         |ner_label         |
+------------------------------+------------------+
|A 3-year-old boy              |patient           |
|autistic disorder             |clinical_condition|
|He                            |patient           |
|illness                       |clinical_event    |
|autistic spectrum disorder    |clinical_condition|
+------------------------------+------------------+
```


#### 5 New Social Determinant of Health Text Classification  Models

We are releasing 5 new models that can be used in Social Determinant of Health related classification tasks.


| model name                                           	            | description                                                                                                                         | predicted entities          |
|------------------------------------------------------------------ |------------------------------------------------------------------------------------------------------------------------------------ |---------------------------- |
| [genericclassifier_sdoh_alcohol_usage_sbiobert_cased_mli](https://nlp.johnsnowlabs.com/2023/01/14/genericclassifier_sdoh_alcohol_usage_sbiobert_cased_mli_en.html)       | This model is intended for detecting alcohol use in clinical notes and trained by using GenericClassifierApproach annotator. | `Present` `Past` `Never` `None`  |
| [genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli](https://nlp.johnsnowlabs.com/2023/01/14/genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli_en.html)| This model is intended for detecting alcohol use in clinical notes and trained by using GenericClassifierApproach annotator.  | `Present` `Never` `None`  |
| [genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli](https://nlp.johnsnowlabs.com/2023/01/14/genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli_en.html)       | This model is intended for detecting tobacco use in clinical notes and trained by using GenericClassifierApproach annotator | `Present` `Past` `Never` `None` |
| [genericclassifier_sdoh_economics_binary_sbiobert_cased_mli](https://nlp.johnsnowlabs.com/2023/01/14/genericclassifier_sdoh_economics_binary_sbiobert_cased_mli_en.html)    | This model classifies related to social economics status in the clinical documents and trained by using GenericClassifierApproach annotator. |  `True` `False` |
| [genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli](https://nlp.johnsnowlabs.com/2023/01/14/genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli_en.html)| This model is intended for detecting substance use in clinical notes and trained by using GenericClassifierApproach annotator. | `Present` `None` |

*Example*:

```python
...
features_asm = FeaturesAssembler()\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("features")

generic_classifier_tobacco = GenericClassifierModel.pretrained("genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["features"])\
    .setOutputCol("class_tobacco")
    
generic_classifier_alcohol = GenericClassifierModel.pretrained("genericclassifier_sdoh_alcohol_usage_sbiobert_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["features"])\
    .setOutputCol("class_alcohol")

text = ["Retired schoolteacher, now substitutes. Lives with wife in location 1439. Has a 27 yo son and a 25 yo daughter. He uses alcohol and cigarettes",
        "The patient quit smoking approximately two years ago with an approximately a 40 pack year history, mostly cigar use.",
        "The patient denies any history of smoking or alcohol abuse. She lives with her one daughter.",
        "She was previously employed as a hairdresser, though says she hasnt worked in 4 years. Not reported by patient, but there is apparently a history of alochol abuse."
      ]
```

*Result*:

```bash
+----------------------------------------------------------------------------------------------------+---------+---------+
|                                                                                                text|  tobacco|  alcohol|
+----------------------------------------------------------------------------------------------------+---------+---------+
|Retired schoolteacher, now substitutes. Lives with wife in location 1439. Has a 27 yo son and a 2...|[Present]|[Present]|
|The patient quit smoking approximately two years ago with an approximately a 40 pack year history...|   [Past]|   [None]|
|        The patient denies any history of smoking or alcohol abuse. She lives with her one daughter.|  [Never]|  [Never]|
|She was previously employed as a hairdresser, though says she hasnt worked in 4 years. Not report...|   [None]|   [Past]|
+----------------------------------------------------------------------------------------------------+---------+---------+
```


#### New `DocumentMLClassifierApproach` Annotator For Training Text Classification Models Using SVM And Logistic Regression Using TfIdf

We have a new `DocumentMLClassifierApproach` that can be used for training text classification models with *Logistic Regression* and *SVM* algorithms. Training data requires "text" and their "label" columns only and the trained model will be a `DocumentMLClassifierModel()`.

Input types: `TOKEN`                  
Output type: `CATEGORY`


| Parameters               | Description                                                      |
|--------------------------|------------------------------------------------------------------|
| labels                   | array to output the label in the original form.                  |
| labelCol                 | column with the value result we are trying to predict.           |
| maxIter                  | maximum number of iterations.                                    |
| tol                      | convergence tolerance after each iteration.                      |
| fitIntercept             | whether to fit an intercept term, default is true.               |
| maxTokenNgram            | the max number of tokens for Ngrams                              |
| minTokenNgram            | the min number of tokens for Ngrams                              |
| vectorizationModelPath   | specify the vectorization model if it has been already trained.  |
| classificationModelPath  | specify the classification model if it has been already trained. |
| classificationModelClass | specify the SparkML classification class; possible values are `logreg`, `svm`  |  


*Example*:

```python
...
classifier_svm= DocumentMLClassifierApproach() \
    .setInputCols("token") \
    .setLabelCol("category") \
    .setOutputCol("prediction") \
    .setMaxTokenNgram(1)\
    .setClassificationModelClass("svm") #or "logreg"

model_svm = Pipeline(stages=[document, token, classifier_svm]).fit(trainingData)

text = [
    ["This 1-year-old child had a gastrostomy placed due to feeding difficulties."], 
    ["He is a pleasant young man who has a diagnosis of bulbar cerebral palsy and hypotonia."], 
    ["The patient is a 45-year-old female whose symptoms are pain in the left shoulder and some neck pain."],
    ["The patient is a 61-year-old female with history of recurrent uroseptic stones."]
]
```

*Result*:

```bash

+----------------------------------------------------------------------------------------------------+----------------+
|text                                                                                                |prediction      |
+----------------------------------------------------------------------------------------------------+----------------+
|He is a pleasant young man who has a diagnosis of bulbar cerebral palsy and hypotonia.              |Neurology       |
|This 1-year-old child had a gastrostomy placed due to feeding difficulties.                         |Gastroenterology|
|The patient is a 61-year-old female with history of recurrent uroseptic stones.                     |Urology         |
|The patient is a 45-year-old female whose symptoms are pain in the left shoulder and some neck pain.|Orthopedic      |
+----------------------------------------------------------------------------------------------------+----------------+

```

####  Option To Return `Deidentification` Output As a Single Document 

We can return `Deidentification()` output as a single document by setting new `setOutputAsDocument` as `True`. If it is `False`, the outputs will be list of sentences as it is used to be.

*Example*:

```python

deid_obfuscated = DeIdentification()\
    .setInputCols(["sentence", "token", "ner_chunk_subentity"]) \
    .setOutputCol("obfuscated") \
    .setMode("obfuscate")\
    .setObfuscateDate(True)\
    .setObfuscateRefFile('obfuscate.txt')\
    .setObfuscateRefSource("file")\
    .setUnnormalizedDateMode("obfuscate")\
    .setOutputAsDocument(True) # or False for sentence level result

text ='''
Record date : 2093-01-13 , David Hale , M.D . , Name : Hendrickson , Ora MR # 7194334 Date : 01/13/93 . Patient : Oliveira, 25 years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street
'''

```

*Result of .setOutputAsDocument(True)*:

```bash

'obfuscated': ['Record date : 2093-01-14 , Beer-Karge , M.D . , Name : Hasan Jacobi Jäckel MR # <MEDICALRECORD> Date : 01-31-1991 . Patient : Herr Anselm Trüb, 51 years-old , Record date : 2080-01-08 . Klinik St. Hedwig . <MEDICALRECORD> Keats Street']

```

*Result of .setOutputAsDocument(False)*:

```bash

'obfuscated': ['Record date : 2093-02-19 , Kaul , M.D . , Name : Frauke Oestrovsky MR # <MEDICALRECORD> Date : 05-08-1971 .',
               'Patient : Lars Bloch, 33 years-old , Record date : 2079-11-11 .',
               'University Hospital of Düsseldorf . <MEDICALRECORD> Keats Street']

```

#### New `Resolution2Chunk` Annotator To Map Entity Resolver Outputs (terminology codes) To Other Clinical Terminologies

We have a new `Resolution2Chunk` annotator that maps the entity resolver outputs to other clinical terminologies.

*Example*:

```python
icd_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented_billable_hcc","en", "clinical/models") \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCol("icd10cm_code")\
    .setDistanceFunction("EUCLIDEAN")
    
resolver2chunk = Resolution2Chunk()\
    .setInputCols(["icd10cm_code"]) \
    .setOutputCol("resolver2chunk")\

chunkerMapper = ChunkMapperModel.pretrained("icd10cm_snomed_mapper", "en", "clinical/models")\
    .setInputCols(["resolver2chunk"])\
    .setOutputCol("mappings")\
    .setRels(["snomed_code"])

sample_text = """Diabetes Mellitus"""
```

*Result*:

```bash
+-----------------+-----------------+------------+-----------+
|text             |ner_chunk        |icd10cm_code|snomed_code|
+-----------------+-----------------+------------+-----------+
|Diabetes Mellitus|Diabetes Mellitus|E109        |170756003  |
+-----------------+-----------------+------------+-----------+
```


#### New `DocMapperModel` Annotator Allows To Use With Any Mapper Model In `DOCUMENT` Type

Any `ChunkMapperModel` can be used with this new annotator called `DocMapperModel` and as its name suggests, it is used to map short strings via `DocumentAssembler` without using any other annotator between to convert strings to `Chunk` type that `ChunkMapperModel` expects. 


*Example*:

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

model = DocMapperModel.pretrained("drug_brandname_ndc_mapper", "en", "clinical/models")\
    .setInputCols("document")\
    .setOutputCol("mappings")

sample_text = "ZYVOX"
```

*Result*:

```bash
| Brand_Name   | Strenth_NDC              |
|:-------------|:-------------------------|
| ZYVOX        | 600 mg/300mL | 0009-4992 |
```


#### Inter-Annotator Agreement (IAA) metrics module that works with NLP Lab seamlessly          

We added a new `get_IAA_metrics()` method to ALAB module. This method allows you to compare and evaluate the annotations in the seed corpus that all annotators annotated the same documents at the begining of an annotation project. It returns all the results in CSV files. Here are the parameters;

- `spark` : SparkSession.
- `conll_dir` (str): path to the folder that conll files in.
- `annotator_names` (list): list of annotator names.
- `set_ref_annotator` (str): reference annotator name. If present, all comparisons made with respect to it, if it is `None` all annotators will be compared by each other. Default is `None`.
- `return_NerDLMetrics` (boolean): If `True`, we get the `full_chunk` and - `partial_chunk_per_token` IAA metrics by using NerDLMetrics. If `False`, we get the chunk based metrics using `evaluate` method of `training_log_parser` module and the token based metrics using classification reports, then write the results in "eval_metric_files" folder. Default is `False`. 
- `save_dir` (str): path to save the token based results dataframes, default is "results_token_based".

For more details and examples, please check [ALAB Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Annotation_Lab/Complete_ALab_Module_SparkNLP_JSL.ipynb).


*Example*:

```python
alab.get_IAA_metrics(spark, conll_dir = path_to_conll_folder, annotator_names = ["annotator_1","annotator_2","annotator_3","annotator_4"], set_ref_annotator = "annotator_1", return_NerDLMetrics = False, save_dir = "./token_based_results")
```


#### Assertion dataset preparation module now supports chunk start and end indices, rather than token indices.

Here are the new features in `get_assertion_data()`;

+ Now it returns the `char_begin` and `char_end` indices of the chunks. These columns can be used in `AssertionDLApproach()` annotator instead of `token_begin` and `token_end` columns for training an Assertion Status Detection model. 
+ Added `included_task_ids` parameter that allows you to prepare the assertion model training dataframe  with only the included tasks. Default is `None`.
+ Added `seed` parameter that allows you to get the same training dataframe at each time when you set `unannotated_label_strategy`. Default is `None`. 

For more details and examples, please check [ALAB Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Annotation_Lab/Complete_ALab_Module_SparkNLP_JSL.ipynb).



#### Added `ner_source` in the `ChunkConverter` Metadata

We added `ner_source` in the metadata of `ChunkConverter` output. In this way, the sources of the chunks can be seen if there are multiple components that have the same NER label in the same pipeline. 

*Example*:

```python
...
age_contextual_parser = ContextualParserApproach() \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("age_cp") \
    .setJsonPath("age.json") \
    .setCaseSensitive(False) \
    .setPrefixAndSuffixMatch(False)    

chunks_age = ChunkConverter()\
    .setInputCols("age_cp")\
    .setOutputCol("age_chunk")
...

sample_text = """The patient is a 28 years old female with a history of gestational diabetes mellitus was diagnosed in April 2002 in County Baptist Hospital ."""
```

*Result*:

```python
[Annotation(chunk, 17, 18, 28, {'tokenIndex': '4', 'entity': 'Age', 'field': 'Age', 'ner_source': 'age_chunk', 'chunk': '0', 'normalized': '', 'sentence': '0', 'confidenceValue': '0.74'})]

```


#### Core Improvements and Bug Fixes

- Added chunk confidence score in the `RelationExtractionModel` metadata
- Added confidence score in the `DocumentLogRegClassifierApproach` metadata
- Fixed non-deterministic Relation Extraction DL Models (30+ models updated in the model hub)
- Fixed incompatible PretrainedPipelines with PySpark v3.2.x and v3.3.x 
- Fixed `ZIP` label issue on `faker` mode with `setZipCodeTag` parameter in `Deidentification` 
- Fixed obfuscated numbers have the same number of chars as the original ones 
- Fixed name obfuscation hashes in `Deidentification` for romanian language 
- Fixed LightPipeline validation parameter for internal annotators
- LightPipeline support for `GenericClassifier` (`FeatureAssembler`)


#### New and Updated Notebooks

+ New [Clinical Text Classification with Spark_NLP Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/30.Clinical_Text_Classification_with_Spark_NLP.ipynb) show how can use medical text with ClassifierDL, MultiClassifierDL, GenericClassifier, and DocumentLogRegClassifier
+ New [Clinical Text Classification with DocumentMLClassifier Notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/30.1.Text_Classification_with_DocumentMLClassifier.ipynb) show how can use medical text with DocumentMLClassifier
+ Updated [ALAB Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Annotation_Lab/Complete_ALab_Module_SparkNLP_JSL.ipynb) with the changes in `get_assertion_data()` and the new `get_IAA_metrics()` method.


#### New and Updated Demos

+ [SOCIAL DETERMINANT](https://demo.johnsnowlabs.com/healthcare/SOCIAL_DETERMINANT/) demo


#### 9 New Clinical Models and Pipelines Added & Updated in Total

+ `ner_oncology_anatomy_general_healthcare`
+ `ner_oncology_biomarker_healthcare`
+ `ner_oncology_unspecific_posology_healthcare`
+ `ner_eu_clinical_case`
+ `genericclassifier_sdoh_economics_binary_sbiobert_cased_mli`
+ `genericclassifier_sdoh_substance_usage_binary_sbiobert_cased_mli`
+ `genericclassifier_sdoh_tobacco_usage_sbiobert_cased_mli`
+ `genericclassifier_sdoh_alcohol_usage_sbiobert_cased_mli`
+ `genericclassifier_sdoh_alcohol_usage_binary_sbiobert_cased_mli`

For all Spark NLP for Healthcare models, please check: [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Healthcare+NLP)


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-healthcare-pagination.html -%}