---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 4.2.4
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_4_2_4
key: docs-licensed-release-notes
modify_date: 2022-12-20
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## 4.2.4

#### Highlights

+ New chunk mapper model for matching drugs by categories as well as other brands and names
+ 4 new NER and classification models for Social Determinant of Health
+ Allow fuzzy matching in the `ChunkMapper` annotator
+ New `NameChunkObfuscatorApproach` annotator to obfuscate doctor and patient names using a custom external list (consistent name obfuscation)
+ New `AssertionChunkConverter` annotator to prepare assertion model training dataset from chunk indices
+ New `training_log_parser` module to parse NER and Assertion Status Detection model training log files
+ Obfuscation of age entities by age groups in `Deidentification`
+ Controlling the behaviour of unnormalized dates while shifting the days in `Deidentification` (`setUnnormalizedDateMode` parameter)
+ Setting default day, months or years for partial dates via `DateNormalizer`
+ Setting label case sensitivity in `AssertionFilterer`
+ `getClasses` method for Zero Shot NER and Zero Shot Relation Extraction models
+ Setting max syntactic distance parameter in `RelationExtractionApproach`
+ Generic Relation Extraction Model (`generic_re`) to extract relations between any named entities using syntactic distances
+ Core improvements and bug fixes
+ New and updated notebooks
+ New and updated demos
    + [MEDICAL QUESTION ANSWERING](https://demo.johnsnowlabs.com/healthcare/MEDICAL_QUESTION_ANSWERING/) 
    + [SMOKING STATUS](https://demo.johnsnowlabs.com/healthcare/SMOKING_STATUS/)
    + [MENTAL HEALTH DEPRESSION](https://demo.johnsnowlabs.com/healthcare/MENTAL_HEALTH_DEPRESSION/)
+ 5 new clinical models and pipelines added & updated in total



#### New Chunk Mapper Model For Matching Drugs by Categories As Well As Other Brands and Names


We have a new `drug_category_mapper` chunk mapper model that maps drugs to their categories, other brands and names. It has two categories called **main category** and **subcategory**.

*Example*:

```python
chunkerMapper = ChunkMapperModel.pretrained("drug_category_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["main_category", "sub_category", "other_name"])\


sample_text= "She is given OxyContin, folic acid, levothyroxine, Norvasc, aspirin, Neurontin."
```

*Result*:

```bash
+-------------+---------------------+-----------------------------------+-----------+
|    ner_chunk|        main_category|                       sub_category|other_names|
+-------------+---------------------+-----------------------------------+-----------+
|    OxyContin|      Pain Management|                  Opioid Analgesics|     Oxaydo|
|   folic acid|         Nutritionals|            Vitamins, Water-Soluble|    Folvite|
|levothyroxine|Metabolic & Endocrine|                   Thyroid Products|     Levo T|
|      Norvasc|       Cardiovascular|                 Antianginal Agents|   Katerzia|
|      aspirin|       Cardiovascular|Antiplatelet Agents, Cardiovascular|        ASA|
|    Neurontin|          Neurologics|                       GABA Analogs|    Gralise|
+-------------+---------------------+-----------------------------------+-----------+
```

#### 4 New NER and Classification Models for Social Determinant of Health

We are releasing 4 new NER and Classification models for Social Determinant of Health.

+ `ner_sdoh_mentions`: Detecting Social Determinants of Health mentions in clinical notes. Predicted entities: `sdoh_community`, `sdoh_economics`, `sdoh_education`, `sdoh_environment`, `behavior_tobacco`, `behavior_alcohol`, `behavior_drug`.

*Example*:

```python
ner_model = MedicalNerModel.pretrained("ner_sdoh_mentions", "en", "clinical/models")\
     .setInputCols(["sentence", "token", "embeddings"])\
     .setOutputCol("ner")

text = """Mr. John Smith is a pleasant, cooperative gentleman with a long standing history (20 years) of diverticulitis. He is married and has 3 children. He works in a bank. He denies any alcohol or intravenous drug use. He has been smoking for many years."""
```

*Result*:

```bash
+----------------+----------------+
|chunk           |ner_label       |
+----------------+----------------+
|married         |sdoh_community  |
|children        |sdoh_community  |
|works           |sdoh_economics  |
|alcohol         |behavior_alcohol|
|intravenous drug|behavior_drug   |
|smoking         |behavior_tobacco|
+----------------+----------------+
```

+ `MedicalBertForSequenceClassification` models that can be used in Social Determinant of Health related classification tasks:

| model name                                           	| description                                                                                                                                        	| predicted entities                          	|
|------------------------------------------------------	|----------------------------------------------------------------------------------------------------------------------------------------------------	|---------------------------------------------	|
| [bert_sequence_classifier_sdoh_community_absent_status]()           	| Classifies the clinical texts related to the loss of social support such as a family member or friend in the clinical documents. A discharge summary was classified True for Community-Absent if the discharge summary had passages related to the loss of social support and False if such passages were not found in the discharge summary.                                                                     	| `True` `False`                	|
| [bert_sequence_classifier_sdoh_community_present_status]() 	| Classifies the clinical texts related to social support such as a family member or friend in the clinical documents. A discharge summary was classified True for Community-Present if the discharge summary had passages related to active social support and False if such passages were not found in the discharge summary.                                                        	| `True` `False`               	|
| [bert_sequence_classifier_sdoh_environment_status]()     	| Classifies the clinical texts related to environment situation such as any indication of housing, homeless or no related passage. A discharge summary was classified as True for the SDOH Environment if there was any indication of housing, False if the patient was homeless and None if there was no related passage.                                                     	| `True` `False` `None`               	|


*Example*:

```python
sequenceClassifier = MedicalBertForSequenceClassification.pretrained("bert_sequence_classifier_sdoh_community_present_status", "en", "clinical/models")\
    .setInputCols(["document","token"])\
    .setOutputCol("class")

sample_text = ["Right inguinal hernia repair in childhood Cervical discectomy 3 years ago Umbilical hernia repair 2137. Retired schoolteacher, now substitutes. Lives with wife in location 1439. Has a 27 yo son and a 25 yo daughter. Name (NI) past or present smoking hx, no EtOH.",
"Atrial Septal Defect with Right Atrial Thrombus Pulmonary Hypertension Obesity, Obstructive Sleep Apnea. Denies tobacco and ETOH. Works as cafeteria worker."]
```

*Result*:

```bash
+----------------------------------------------------------------------------------------------------+-------+
|                                                                                                text| result|
+----------------------------------------------------------------------------------------------------+-------+
|Right inguinal hernia repair in childhood Cervical discectomy 3 years ago Umbilical hernia repair...| [True]|
|Atrial Septal Defect with Right Atrial Thrombus Pulmonary Hypertension Obesity, Obstructive Sleep...|[False]|
+----------------------------------------------------------------------------------------------------+-------+
```



#### Allow Fuzzy Matching in the `ChunkMapper` Annotator

There are multiple options to achieve fuzzy matching using the ChunkMapper annotation:
- **Partial Token NGram Fingerprinting**: Useful to combine two frequent usecases; when there are noisy non informative tokens at the beginning / end of the chunk and the order of the chunk is not absolutely relevant. i.e. stomach acute pain --> acute pain stomach ; metformin 100 mg --> metformin.
- **Char NGram Fingerprinting**: Useful in usecases that involve typos or different spacing patterns for chunks. i.e. head ache / ache head --> headache ; metformini / metformoni / metformni --> metformin
- **Fuzzy Distance (Slow)**: Useful when the mapping can be defined in terms of edit distance thresholds using functions like char based like *Levenshtein*, *Hamming*, *LongestCommonSubsequence* or token based like *Cosine*, *Jaccard*.

The mapping logic will be run in the previous order also ordering by longest key inside each option as an intuitive way to minimize false positives.

*Basic Mapper Example*:
```bash
cm = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setLowerCase(True) \
        .setRels(["action", "treatment"]) \


text = """The patient was given Lusa Warfarina 5mg and amlodipine 10 MG. 
The patient was given Aspaginaspa, coumadin 5 mg, coumadin, and he has metamorfin"""


# Since mappers only match one-to-one

| ner_chunk          | fixed_chunk | action                | treatment    |
|:-------------------|:------------|:----------------------|:-------------|
| Aspaginaspa        | nan         | nan                   | nan          |
| Lusa Warfarina 5mg | nan         | nan                   | nan          |
| amlodipine 10      | nan         | nan                   | nan          |
| coumadin           | coumadin    | Coagulation Inhibitor | hypertension |
| coumadin 5 mg      | nan         | nan                   | nan          |
| metamorfin         | nan         | nan                   | nan          |
```
Since mappers only match one-to-one, we see that only 1 chunk has action and teatment in the table above.


*Token Fingerprinting Example*:

```python
cm = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setLowerCase(True) \
        .setRels(["action", "treatment"]) \
        .setAllowMultiTokenChunk(True) \
        .setEnableTokenFingerprintMatching(True) \
        .setMinTokenNgramFingerprint(1) \
        .setMaxTokenNgramFingerprint(3) \
        .setMaxTokenNgramDroppingCharsRatio(0.5)
```

*Result*:

```bash
| ner_chunk           | fixed_chunk     | action                 | treatment    |
|:--------------------|:----------------|:-----------------------|:-------------|
| Aspaginaspa         | nan             | nan                    | nan          |
| Lusa Warfarina 5mg  | Warfarina lusa  | Analgesic              | diabetes     |
| amlodipine 10       | amlodipine      | Calcium Ions Inhibitor | hypertension |
| coumadin            | coumadin        | Coagulation Inhibitor  | hypertension |
| coumadin 5 mg       | coumadin        | Coagulation Inhibitor  | hypertension |
| metamorfin          | nan             | nan                    | nan          |
```

*Token and Char Fingerprinting Example*:

```python
cm = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setLowerCase(True) \
        .setRels(["action", "treatment"]) \
        .setAllowMultiTokenChunk(True) \
        .setEnableTokenFingerprintMatching(True) \
        .setMinTokenNgramFingerprint(1) \
        .setMaxTokenNgramFingerprint(3) \
        .setMaxTokenNgramDroppingCharsRatio(0.5) \
        .setEnableCharFingerprintMatching(True) \
        .setMinCharNgramFingerprint(1) \
        .setMaxCharNgramFingerprint(3)
```

*Result*:

```bash
| ner_chunk           | fixed_chunk    | action                  | treatment    |
|:--------------------|:---------------|:------------------------|:-------------|
| Aspaginaspa         | aspagin        | Cycooxygenase Inhibitor | arthritis    |
| Lusa Warfarina 5mg  | Warfarina lusa | Analgesic               | diabetes     |
| amlodipine 10       | amlodipine     | Calcium Ions Inhibitor  | hypertension |
| coumadin            | coumadin       | Coagulation Inhibitor   | hypertension |
| coumadin 5 mg       | coumadin       | Coagulation Inhibitor   | hypertension |
| metamorfin          | nan            | nan                     | nan          |
```

*Token and Char Fingerprinting With Fuzzy Distance Calculation Example*:

```python
cm = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("mappings.json") \
        .setLowerCase(True) \
        .setRels(["action"]) \
        .setAllowMultiTokenChunk(True) \
        .setEnableTokenFingerprintMatching(True) \
        .setMinTokenNgramFingerprint(1) \
        .setMaxTokenNgramFingerprint(3) \
        .setMaxTokenNgramDroppingCharsRatio(0.5) \
        .setEnableCharFingerprintMatching(True) \
        .setMinCharNgramFingerprint(1) \
        .setMaxCharNgramFingerprint(3) \
        .setEnableFuzzyMatching(True) \
        .setFuzzyMatchingDistanceThresholds(0.31)
```

*Result*:

```bash
| ner_chunk          | fixed_chunk    | action                  | treatment    |
|:-------------------|:---------------|:------------------------|:-------------|
| Aspaginaspa        | aspagin        | Cycooxygenase Inhibitor | arthritis    |
| Lusa Warfarina 5mg | Warfarina lusa | Analgesic               | diabetes     |
| amlodipine 10      | amlodipine     | Calcium Ions Inhibitor  | hypertension |
| coumadin           | coumadin       | Coagulation Inhibitor   | hypertension |
| coumadin 5 mg      | coumadin       | Coagulation Inhibitor   | hypertension |
| metamorfin         | metformin      | hypoglycemic            | diabetes     |
```


You can check [Chunk_Mapping](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb) notebook for more examples.



#### New `NameChunkObfuscatorApproach` Annotator to Obfuscate Doctor and Patient Names Using a Custom External List (consistent name obfuscation)

We have a new `NameChunkObfuscatorApproach` annotator that can be used in deidentification tasks for replacing *doctor* and *patient* names with *fake names* using a reference document.

*Example*:

```python
names = """Mitchell#NAME
Jackson#NAME
Leonard#NAME
Bowman#NAME
Fitzpatrick#NAME
Melody#NAME"""

with open('names_test.txt', 'w') as file:
    file.write(names)

nameChunkObfuscator = NameChunkObfuscatorApproach()\
  .setInputCols("ner_chunk")\
  .setOutputCol("replacement")\
  .setRefFileFormat("csv")\
  .setObfuscateRefFile("names_test.txt")\
  .setRefSep("#")\

text = '''John Davies is a 62 y.o. patient admitted. Mr. Davies was seen by attending physician Dr. Lorand and was scheduled for emergency assessment. '''
```

*Result*:

```bash
Original text   :  John Davies is a 62 y.o. patient admitted. Mr. Davies was seen by attending physician Dr. Lorand and was scheduled for emergency assessment.

Obfuscated text :  Fitzpatrick is a <AGE> y.o. patient admitted. Mr. Bowman was seen by attending physician Dr. Melody and was scheduled for emergency assessment.
```

You can check [Clinical DeIdentification](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb) notebook for more examples.



#### New `AssertionChunkConverter` Annotator to Prepare Assertion Model Training Dataset From Chunk Indices

In some cases, there may be issues while creating the chunk column by using token indices and losing some data while training and testing the assertion status model if there are issues in these token indices. So we developed a new `AssertionChunkConverter` annotator that takes **begin and end indices of the chunks** as input and creates an extended chunk column with metadata that can be used for assertion status detection model training.

*Example*:

```python
...
converter = AssertionChunkConverter() \
    .setInputCols("tokens")\
    .setChunkTextCol("target")\
    .setChunkBeginCol("char_begin")\
    .setChunkEndCol("char_end")\
    .setOutputTokenBeginCol("token_begin")\
    .setOutputTokenEndCol("token_end")\
    .setOutputCol("chunk")

sample_data = spark.createDataFrame([["An angiography showed bleeding in two vessels off of the Minnie supplying the sigmoid that were succesfully embolized.", "Minnie", 57, 63],
     ["After discussing this with his PCP, Leon was clear that the patient had had recurrent DVTs and ultimately a PE and his PCP felt strongly that he required long-term anticoagulation ", "PCP", 31, 34]])\
     .toDF("text", "target", "char_begin", "char_end")
```

*Result*:

```bash
+------+----------+--------+-----------+---------+--------------------------+------------------------+------+----------------------------------------------+
|target|char_begin|char_end|token_begin|token_end|tokens[token_begin].result|tokens[token_end].result|target|chunk                                         |
+------+----------+--------+-----------+---------+--------------------------+------------------------+------+----------------------------------------------+
|Minnie|57        |62      |10         |10       |Minnie                    |Minnie                  |Minnie|[{chunk, 57, 63, Minnie, {sentence -> 0}, []}]|
|PCP   |31        |34      |5          |5        |PCP                       |PCP                     |PCP   |[{chunk, 31, 33, PCP, {sentence -> 0}, []}]   |
+------+----------+--------+-----------+---------+--------------------------+------------------------+------+----------------------------------------------+
```


#### New `training_log_parser` Module to Parse Training Log Files of NER And Assertion Status Detection Models  

We are releasing a new `training_log_parser` module that helps to parse NER and Assertion Status Detection model training log files using a single module. Here are the methods and their descriptions:

||Description|ner_log_parser|assertion_log_parser|
|-|-|-|-|
|How to import | You can import this module for NER and Assertion as shown here |`from sparknlp_jsl.utils.training_log_parser import ner_log_parser`   | `from sparknlp_jsl.utils.training_log_parser import assertion_log_parser`  |
| `get_charts`  | Plots the figures of metrics ( precision, recall, f1) vs epochs  | `ner_log_parser.get_charts(log_file, threshold)`  | `assertion_log_parser.get_charts(log_file, labels, threshold)`  |  
|`loss_plot` | Plots the figures of validation and test loss values vs epochs.  | `ner_log_parser.loss_plot(path)`  | `assertion_log_parser.loss_plot(path)`  |  
| `get_best_f1_scores`  | Returns the best Micro and Macro F1 Scores on test set  | `ner_log_parser.get_best_f1_scores(path)`   | `assertion_log_parser.get_best_f1_scores(path)`  |
| `parse_logfile`  | Returns the parsed log file in pandas dataframe format with the order of *label-score dataframe, epoch-metrics dataframe and graph file used in tranining*.  | `ner_log_parser.parse_logfile(path)`  | `assertion_log_parser.parse_logfile(path, labels)`  |
|`evaluate` | if verbose, returns overall performance, as well as **performance per chunk type**; otherwise, simply returns overall precision, recall, f1 scores. Ground truth and predictions should be provided in pandas dataframe.   | `ner_log_parser.evaluate(preds_df['ground_truth'].values, preds_df['prediction'].values)`  | - |

*Import*

```python
from sparknlp_jsl.utils.training_log_parser import ner_log_parser, assertion_log_parser

ner_parser = ner_log_parser()
assertion_parser = assertion_log_parser()
```

*Example for NER loss_plot method*:

```python
ner_parser.loss_plot('NER_training_log_file.log')
```

*Result*:

![image](https://user-images.githubusercontent.com/76607915/208672630-2c65f533-f3cf-432b-9290-8eeac4aaa083.png)

*Example for NER evaluate method*:

```python
metrics = ner_parser.evaluate(preds_df['ground_truth'].values, preds_df['prediction'].values)
```

*Result*:

![image](https://user-images.githubusercontent.com/76607915/208672122-41cbf3e0-2f38-47b9-9f65-47503de38302.png)

![image](https://user-images.githubusercontent.com/76607915/208672159-1ff7733c-6dba-45e6-a0de-ba801565a564.png)

*Example for Assertion get_best_f1_scores method*:

```python
assertion_parser.get_best_f1_scores('Assertion_training_log_file.log', ['Absent', 'Present'])
```

*Result*:

![image](https://user-images.githubusercontent.com/76607915/208671659-eca0b063-234b-412f-bfac-d2da8625a39b.png)



#### Obfuscation of Age Entities by Age Groups in `Deidentification`

We have a new `setAgeRanges()` parameter in `Deidentification` annotator that provides the ability to set a custom range for obfuscation of `AGE` entities by another `age` within that age group (range). Default age groups list is `[1, 4, 12, 20, 40, 60]` and users can set any range.

 - Infant = 0-1 year.
 - Toddler = 2-4 yrs.
 - Child = 5-12 yrs.
 - Teen = 13-19 yrs.
 - Adult = 20-39 yrs.
 - Middle Age Adult = 40-59 yrs.
 - Senior Adult = 60+

*Example*:

```python
deidentification = DeIdentification()\
    .setInputCols(["sentence", "token", "age_chunk"]) \
    .setOutputCol("obfuscation") \
    .setMode("obfuscate")\
    .setObfuscateDate(True)\
    .setObfuscateRefSource("faker") \
    .setAgeRanges([1, 4, 12, 20, 40, 60, 80])
```

*Result*:

```bash
+--------------------------------+---------+--------------------------------+
|text                            |age_chunk|obfuscation                     |
+--------------------------------+---------+--------------------------------+
|1 year old baby                 |1        |2 year old baby                 |
|4 year old kids                 |4        |6 year old kids                 |
|A 15 year old female with       |15       |A 12 year old female with       |
|Record date: 2093-01-13, Age: 25|25       |Record date: 2093-03-01, Age: 30|
|Patient is 45 years-old         |45       |Patient is 44 years-old         |
|He is 65 years-old male         |65       |He is 75 years-old male         |
+--------------------------------+---------+--------------------------------+
```


#### Controlling the behaviour of unnormalized dates while shifting the days in `Deidentification` (`setUnnormalizedDateMode` parameter)

Two alternatives can be used when deidentification in unnormalized date formats, these are mask and obfuscation.
- `setUnnormalizedDateMode('mask')` parameter is used to **mask** the DATE entities that can not be normalized.
- `setUnnormalizedDateMode('obfuscate')` parameter is used to **obfuscate** the DATE entities that can not be normalized.

*Example*:

```python
de_identification = DeIdentification() \
    .setInputCols(["ner_chunk", "token", "document2"]) \
    .setOutputCol("deid_text") \
    .setMode("obfuscate") \
    ...
    .setUnnormalizedDateMode("mask") # or obfuscation
```

*Result*:

```bash
+-----------+---------+------------+------------+
|text       |dateshift| mask       | obfuscation|
+-----------+---------+------------+------------+
|04/19/2018 |-5       | 04/14/2018 | 04/14/2018 |
|04-19-2018 |-2       | 04-17-2018 | 04-17-2018 |
|19 Apr 2018|10       | <DATE>     | 10-10-1975 |
|04-19-18   |20       | <DATE>     | 03-23-2001 |
+-----------+---------+------------+------------+
```


#### Setting Default Day, Months or Years for Partial Dates via `DateNormalizer`

We have 3 new parameters to make `DateNormalizer` more flexible with date replacing. If any of the day, month and year information is missing in the date format, the following default values will be added.

- `setDefaultReplacementDay`: default value is 15
- `setDefaultReplacementMonth`: default value is July or 6
- `setDefaultReplacementYear`:  default value is 2020


*Example*:

```python
date_normalizer_us = DateNormalizer()\
    .setInputCols('date_chunk')\
    .setOutputCol('normalized_date_us')\
    .setOutputDateformat('us')\
    .setDefaultReplacementDay("15")\
    .setDefaultReplacementMonth("6")\
    .setDefaultReplacementYear("2020")
```

*Result*:

```bash
+------------+------------+------------------+
|text        |date_chunk  |normalized_date_us|
+------------+------------+------------------+
|08/02/2018  |08/02/2018  |08/02/2018        |
|3 April 2020|3 April 2020|04/03/2020        |
|03/2021     |03/2021     |03/15/2021        |
|05 Jan      |05 Jan      |01/05/2020        |
|01/05       |01/05       |01/05/2020        |
|2022        |2022        |06/15/2022        |
+------------+------------+------------------+
```

You can check [Date Normalizer](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/25.Date_Normalizer.ipynb) notebook for more examples



#### Setting Label Case Sensitivity in `AssertionFilterer`

We have case sensitive filtering flexibility for labels by setting new `setCaseSensitive(True)` in `AssertionFilterer` annotator.

*Example*:

```python
assertion_filterer = AssertionFilterer()\
    .setInputCols("sentence","ner_chunk","assertion")\
    .setOutputCol("assertion_filtered")\
    .setCaseSensitive(False)\
    .setWhiteList(["ABsent"])

sample_text = "The patient was admitted 2 weeks ago with a headache. No alopecia was noted."
```

*Result*:

```bash
| chunks   | entities                  | assertion | confidence |
| -------- | ------------------------- | --------- | ---------- |
| Alopecia | Disease_Syndrome_Disorder | Absent    |          1 |
```


#### `getClasses` Method to Zero Shot NER and Zero Shot Relation Extraction Models

The predicted entities of `ZeroShotNerModel` and `ZeroShotRelationExtractionModels` can be extracted with `getClasses` methods just like NER annotators.

*Example*:

```python
zero_shot_ner = ZeroShotNerModel.pretrained("zero_shot_ner_roberta", "en", "clinical/models")\
    .setEntityDefinitions({
            "PROBLEM": ["What is the disease?", "What is the problem?" ,"What does a patient suffer"],
            "DRUG": ["Which drug?", "Which is the drug?", "What is the drug?"],
            "ADMISSION_DATE": ["When did patient admitted to a clinic?"],
            "PATIENT_AGE": ["How old is the patient?",'What is the gae of the patient?']  })\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("zero_shot_ner")

zero_shot_ner.getClasses()
```

*Result*:

```bash
['DRUG', 'PATIENT_AGE', 'ADMISSION_DATE', 'PROBLEM']
```


#### Setting Max Syntactic Distance Flexibility In `RelationExtractionApproach`

Now we are able to set maximal syntactic distance as threshold in `RelationExtractionApproach` while training relation extraction models.

```python
reApproach = RelationExtractionApproach()\
    .setInputCols(["embeddings", "pos_tags", "train_ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setLabelColumn("rel")\
    ...
    .setMaxSyntacticDistance(10)
```


#### Generic Relation Extraction Model (`generic_re`) to extract relations between any named entities using syntactic distances

We already have more than 80 relation extraction (RE) models that can extract relations between certain named entities. Nevertheless, there are some rare entities or cases that you may not find the right RE or the one you find may not work as expected due to nature of your dataset. In order to ease this burden, we are releasing a generic RE model (`generic_re`) that can be used between any named entities using the syntactic distances, POS tags and dependency tree between the entities. You can tune this model by using the `setMaxSyntacticDistance` param.

*Example*:

```python
reModel = RelationExtractionModel()\
    .pretrained("generic_re")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setRelationPairs(["Biomarker-Biomarker_Result", "Biomarker_Result-Biomarker", "Oncogene-Biomarker_Result", "Biomarker_Result-Oncogene", "Pathology_Test-Pathology_Result", "Pathology_Result-Pathology_Test"]) \
    .setMaxSyntacticDistance(4)
    
text = """Pathology showed tumor cells, which were positive for estrogen and progesterone receptors."""    
```
*Result*:

```bash
|sentence |entity1_begin |entity1_end | chunk1    | entity1          |entity2_begin |entity2_end | chunk2                 | entity2          | relation                        |confidence|
|--------:|-------------:|-----------:|:----------|:-----------------|-------------:|-----------:|:-----------------------|:-----------------|:--------------------------------|----------|
|       0 |            1 |          9 | Pathology | Pathology_Test   |           18 |         28 | tumor cells            | Pathology_Result | Pathology_Test-Pathology_Result |         1|
|       0 |           42 |         49 | positive  | Biomarker_Result |           55 |         62 | estrogen               | Biomarker        | Biomarker_Result-Biomarker      |         1|
|       0 |           42 |         49 | positive  | Biomarker_Result |           68 |         89 | progesterone receptors | Biomarker        | Biomarker_Result-Biomarker      |         1|
```

#### Core improvements and bug fixes

- Fixed obfuscated addresses capitalized word style
- Added more patterns for Date Obfuscation  
- Improve speed of `get_conll_data()` method in alab module
- Fixed serialization Issue with MLFlow ContextualParser
- Renamed `TFGraphBuilder.setIsMedical` to `TFGraphBuilder.setIsLicensed`


#### New and Updated Notebooks

+ Updated [ZeroShot Clinical NER Notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.6.ZeroShot_Clinical_NER.ipynb) with `getClasses` method for zero shot NER models.

+ Updated [Clinical Assertion Notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/2.Clinical_Assertion_Model.ipynb) with `AssertionChunkConverter`, `AssertionFilterer` and  `TFGraphBuilder.setIsLicensed` examples.

+ Updated [Clinical Entity Resolvers Notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb) with `AssertionFilterer` example.


+ Updated [Clinical DeIdentification Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb) with `setUnnormalizedDateMode` and  `NameChunkObfuscatorApproach` example.

+ Updated [ZeroShot Clinical Relation Extraction Notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.3.ZeroShot_Clinical_Relation_Extraction.ipynb) with `getClasses` and `setMaxSyntacticDistance` method for Relation Extraction models.

+ Updated [Date Normalizer](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/25.Date_Normalizer.ipynb) notebook with `DateNormalizer` for dynamic date replace values.

+ Updated [Chunk Mapping](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb) notebook with fuzzy matching flexibility examples.


#### New and Updated Demos

+ [MEDICAL QUESTION ANSWERING](https://demo.johnsnowlabs.com/healthcare/MEDICAL_QUESTION_ANSWERING/) 
+ [SMOKING STATUS](https://demo.johnsnowlabs.com/healthcare/SMOKING_STATUS/)
+ [MENTAL HEALTH DEPRESSION](https://demo.johnsnowlabs.com/healthcare/MENTAL_HEALTH_DEPRESSION/)



#### 5 New Clinical Models and Pipelines Added & Updated in Total

+ `drug_category_mapper`
+ `ner_sdoh_mentions`
+ `bert_sequence_classifier_sdoh_community_absent_status`
+ `bert_sequence_classifier_sdoh_community_present_status`
+ `bert_sequence_classifier_sdoh_environment_status`


For all Spark NLP for healthcare models, please check: [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Healthcare+NLP)


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-healthcare-pagination.html -%}