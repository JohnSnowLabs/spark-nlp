---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 4.2.1
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_4_2_1
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## 4.2.1

#### Highlights

+ Creating new chunks with `NerConverterInternal` by merging chunks by skipping stopwords in between.
+ Adding relation direction to `RelationExtraction` models to make the relations direction-aware. 
+ Using proper regional date formats in the `DeIdentification` module.
+ Being able to play with different date formats in `DateNormalizer` output.
+ New `Replacer` annotator to replace chunks with their normalized versions (`DateNormalizer') in documents.
+ New `ModelTracer` helper class to generate and add model UID and timestamps of the stages in a pipeline
+ Added entity source and labels to the `AssertionFilter` metadata
+ New chunk mapper and sentence entity resolver models and a pipeline for CVX
+ Updated clinical NER models with new labels
+ New Certification Training notebooks for the `johnsnowlabs` library
+ New and updated notebooks
+ 6 new clinical models and pipelines added & updated in total

</div><div class="h3-box" markdown="1">

#### Creating New Chunks with `NerConverterInternal` by Merging Chunks by Skipping Stopwords in Between.

`NerConverterInternal`'s new `setIgnoreStopWords` parameter allows merging between chunks with the same label, ignoring stopwords and punctuations.

```python
txt = """ The qualified manufacturers for this starting material are:
Alpha Chemicals Pvt LTD
17, R K Industry House, Walbhat Rd, Goregaon – 400063
Mumbai, Maharashtra, India
Beta Chemical Co., Ltd
Huan Cheng Xi Lu 3111hao Hai Guan Da Ting
Shanghai, China """
```
*Example for default:*

```python
NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner_deid"])\
    .setOutputCol("chunk_deid")\
    .setGreedyMode(True)\
    .setWhiteList(['LOCATION'])
```

*Results:*

```bash
| chunks                   | entities | begin | end |
|:-------------------------|:---------|------:|----:|
| R K Industry House       | LOCATION |    90 | 107 |
| Walbhat                  | LOCATION |   110 | 116 |
| Mumbai                   | LOCATION |   141 | 146 |
| Maharashtra              | LOCATION |   149 | 159 |
| India                    | LOCATION |   162 | 166 |
| Huan Cheng Xi Lu 3111hao | LOCATION |   191 | 214 |
| Shanghai                 | LOCATION |   234 | 241 |
| China                    | LOCATION |   244 | 248 |
```

*Example for setting setIgnoreStopWords parameter:*

```python
NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner_deid"])\
    .setOutputCol("chunk_deid")\
    .setGreedyMode(True)\
    .setWhiteList(['LOCATION'])\
    .setIgnoreStopWords(['\n', ',', "and", 'or', '.'])
```

*Results:*

```bash
| chunks                     | entities | begin | end |
|:---------------------------|:---------|------:|----:|
| R K Industry House Walbhat | LOCATION |    90 | 116 |
| Mumbai Maharashtra India   | LOCATION |   141 | 166 |
| Huan Cheng Xi Lu 3111hao   | LOCATION |   191 | 214 |
| Shanghai China             | LOCATION |   234 | 248 |
```

</div><div class="prev_ver h3-box" markdown="1">

#### Adding Relation Direction to  `RelationExtraction` Models to Make the Relations Direction-aware.

We have a new `setRelationDirectionCol` parameter that is used during training with a new separate column that specified relationship directions. The column should contain one of the following values:

 - `rightwards`: The first entity in the text is also the first argument of the relation (as well as the second entity in the text is the second argument). In other words, the relation arguments are ordered *left to right* in the text.
 - `leftwards`: The first entity in the text is the second argument of the relation (and the second entity in the text is the first argument).
 - `both`: Order doesn't matter (relation is symmetric).

In our test cases, it was observed that **the accuracy increased significantly** when we just add `setRelationDirectionCol` parameter by keeping the other parameter as they are.

*Example:*

```bash
+--------------------+---------+---------+--------------------+----+----------+
|              chunk1|   label1|   label2|              chunk2| rel|   rel_dir|
+--------------------+---------+---------+--------------------+----+----------+
|expected long ter...|treatment|treatment|         a picc line|   O|      both|
|    light-headedness|  problem|  problem|         diaphoresis| PIP|rightwards|
| po pain medications|treatment|  problem|            his pain|TrAP| leftwards|
|bilateral pleural...|  problem|  problem|increased work of...| PIP|rightwards|
|    her urine output|     test|  problem|           decreased|TeRP|rightwards|
|his psychiatric i...|  problem|  problem|his neurologic in...| PIP|rightwards|
|   white blood cells|     test|     test|     red blood cells|   O|      both|
|            chloride|     test|     test|                 bun|   O|      both|
|     further work-up|     test|  problem|his neurologic co...|TeCP|rightwards|
|         four liters|treatment|     test|      blood pressure|   O|      both|
+--------------------+---------+---------+--------------------+----+----------+
```

```python
re_approach_with_dir = RelationExtractionApproach()\
    .setInputCols(["embeddings", "pos_tags", "train_ner_chunks", "dependencies"])\
    .setOutputCol("relations")\
    .setLabelColumn("rel")\
    ...
    .setRelationDirectionCol("rel_dir")
```

</div><div class="prev_ver h3-box" markdown="1">

#### Using Proper Regional date Formats in  `DeIdentification` Module

You can specify the format for date entities that will be shifted to the new date or converted to a year.

```python
de_identification = DeIdentification() \
    .setInputCols(["ner_chunk", "token", "sentence"]) \
    .setOutputCol("dei_id") \
    .setRegion('us') # 'eu' for Europe
```

</div><div class="prev_ver h3-box" markdown="1">

#### Being Able to Play With Different Date Formats in `DateNormalizer` Output

Now we can customize the normalized date formats in the output of `DateNormalizer` by using the new `setOutputDateformat` parameter. There are two options to do that; `us` for `MM/DD/YYYY`, `eu` for `DD/MM/YYYY` formats.

*Example:*

```python
date_normalizer_us = DateNormalizer()\
    .setInputCols('date_chunk')\
    .setOutputCol('normalized_date_us')\
    .setOutputDateformat('us')

date_normalizer_eu = DateNormalizer()\
    .setInputCols('date_chunk')\
    .setOutputCol('normalized_date_eu')\
    .setOutputDateformat('eu')

sample_text = ['She was last seen in the clinic on Jan 30, 2018, by Dr. Y.',
               'Chris Brown was discharged on 12Mar2021',
               'We reviewed the pathology obtained on 13.04.1999.']
```

*Results:*

```bash
+----------------------------------------------------------+------------+------------------+------------------+
|text                                                      |date_chunk  |normalized_date_eu|normalized_date_us|
+----------------------------------------------------------+------------+------------------+------------------+
|She was last seen in the clinic on Jan 30, 2018, by Dr. Y.|Jan 30, 2018|30/01/2018        |01/30/2018        |
|Chris Brown was discharged on 12Mar2021                   |12Mar2021   |12/03/2021        |03/20/2021        |
|We reviewed the pathology obtained on 13.04.1999.         |13.04.1999  |13/04/1999        |04/13/1999        |
+----------------------------------------------------------+------------+------------------+------------------+
```

</div><div class="prev_ver h3-box" markdown="1">

#### New `Replacer` Annotator To Replace Chunks With Their Normalized Versions (`DateNormalizer`) In Documents

We have a new `Replacer` annotator that returns the original document by replacing it with the normalized version of the original chunks.

*Example:*

```python
date_normalizer = DateNormalizer()\
    .setInputCols('date_chunk')\
    .setOutputCol('normalized_date')\

replacer = Replacer()\
    .setInputCols(["normalized_date","document"])\
    .setOutputCol("replaced_document")

sample_text = ['She was last seen in the clinic on Jan 30, 2018, by Dr. Y.',
               'Chris Brown was discharged on 12Mar2021',
               'We reviewed the pathology obtained on 13.04.1999.']
```

*Results:*

```bash
+----------------------------------------------------------+---------------+--------------------------------------------------------+
|text                                                      |normalized_date|replaced_document                                       |
+----------------------------------------------------------+---------------+--------------------------------------------------------+
|She was last seen in the clinic on Jan 30, 2018, by Dr. Y.|2018/01/30     |She was last seen in the clinic on 2018/01/30, by Dr. Y.|
|Chris Brown was discharged on 12Mar2021                   |2021/03/12     |Chris Brown was discharged on 2021/03/12                |
|We reviewed the pathology obtained on 13.04.1999.         |1999/04/13     |We reviewed the pathology obtained on 1999/04/13.       |
+----------------------------------------------------------+---------------+--------------------------------------------------------+
```
</div><div class="prev_ver h3-box" markdown="1">

#### New `ModelTracer` Helper Class to Generate and Add Model UID and Timestamps of the Stages in a Pipeline

`ModelTracer` allows to track the UIDs and timestamps of each stage of a pipeline.

*Example:*

```python
from sparknlp_jsl.modelTracer import ModelTracer
...
pipeline = Pipeline(
    stages=[
        documentAssembler,
        tokenizer,
        tokenClassifier,
        ])

df = pipeline.fit(data).transform(data)

result = ModelTracer().addUidCols(pipeline = pipeline, df = df)
result.show(truncate=False)
```

*Results:*

```bash
+----+--------+-----+---+----------------------------------------------------------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------+
|text|document|token|ner|documentassembler_model_uid                                           |tokenizer_model_uid                                           |bert_for_token_classification_model_uid                                           |
+----+--------+-----+---+----------------------------------------------------------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------+
|... |...     |...  |...|{uid -> DocumentAssembler_a666efd1d789, timestamp -> 2022-10-21_11:34}|{uid -> Tokenizer_01fbad79f069, timestamp -> 2022-10-21_11:34}|{uid -> BERT_FOR_TOKEN_CLASSIFICATION_675a6a750b89, timestamp -> 2022-10-21_11:34}|
+----+--------+-----+---+----------------------------------------------------------------------+--------------------------------------------------------------+----------------------------------------------------------------------------------+
```

</div><div class="prev_ver h3-box" markdown="1">

#### Added Entity Source and Labels to the `AssertionFilter` Metadata

Now the `AssertionFilter` annotator returns the entity source and assertion labels in the metadata.

*Example:*
```python
assertionFilterer = AssertionFilterer() \
    .setInputCols(["sentence","ner_chunk","assertion"]) \
    .setOutputCol("filtered") \
    .setCriteria("assertion") \
    .setWhiteList(["Absent"])
text = "Patient has a headache for the last 2 weeks, no alopecia noted."
```
*Results:*
```bash
# before v4.2.1
+-----------------------------------------------------------------------------------------------------+
|filtered                                                                                             |
+-----------------------------------------------------------------------------------------------------+
|[{chunk, 48, 55, alopecia, {entity -> PROBLEM, sentence -> 0, chunk -> 1, confidence -> 0.9988}, []}]|
+-----------------------------------------------------------------------------------------------------+

# v4.2.1
+---------------------------------------------------------------------------------------------------------------------------------------------------+
|filtered                                                                                                                                           |
+---------------------------------------------------------------------------------------------------------------------------------------------------+
|[{chunk, 48, 55, alopecia, {chunk -> 1, confidence -> 0.9987, ner_source -> ner_chunk, assertion -> Absent, entity -> PROBLEM, sentence -> 0}, []}]|
+---------------------------------------------------------------------------------------------------------------------------------------------------+
```

</div><div class="prev_ver h3-box" markdown="1">

#### New Chunk Mapper and Sentence Entity Resolver Models And A Pipeline for CVX  

+ We are releasing 2 new chunk mapper models to map entities to their corresponding CVX codes, vaccine names and CPT codes. There are 3 types of vaccine names mapped; `short_name`, `full_name` and `trade_name`

| model name      | description                                                                             |
|-----------------|-----------------------------------------------------------------------------------------|
| [cvx_name_mapper](https://nlp.johnsnowlabs.com/2022/10/12/cvx_name_mapper_en.html) | Mapping vaccine products to their corresponding CVX codes, vaccine names and CPT codes. |
| [cvx_code_mapper](https://nlp.johnsnowlabs.com/2022/10/12/cvx_code_mapper_en.html) | Mapping CVX codes to their corresponding vaccine names and CPT codes.          |


*Example:*

```python
chunkerMapper = ChunkMapperModel\
    .pretrained("cvx_name_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["cvx_code", "short_name", "full_name", "trade_name", "cpt_code"])

data = spark.createDataFrame([['DTaP'], ['MYCOBAX'], ['cholera, live attenuated']]).toDF('text')
```

*Results:*

```bash
+--------------------------+--------+--------------------------+-------------------------------------------------------------+------------+--------+
|chunk                     |cvx_code|short_name                |full_name                                                    |trade_name  |cpt_code|
+--------------------------+--------+--------------------------+-------------------------------------------------------------+------------+--------+
|[DTaP]                    |[20]    |[DTaP]                    |[diphtheria, tetanus toxoids and acellular pertussis vaccine]|[ACEL-IMUNE]|[90700] |
|[MYCOBAX]                 |[19]    |[BCG]                     |[Bacillus Calmette-Guerin vaccine]                           |[MYCOBAX]   |[90585] |
|[cholera, live attenuated]|[174]   |[cholera, live attenuated]|[cholera, live attenuated]                                   |[VAXCHORA]  |[90625] |
+--------------------------+--------+--------------------------+-------------------------------------------------------------+------------+--------+
```

+ `sbiobertresolve_cvx`: This sentence entity resolver model maps vaccine entities to CVX codes using `sbiobert_base_cased_mli` Sentence Bert Embeddings. Additionally, this model returns status of the vaccine (Active/Inactive/Pending/Non-US) in `all_k_aux_labels` column.

*Example:*

```python
cvx_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_cvx", "en", "clinical/models")\
    .setInputCols(["ner_chunk", "sbert_embeddings"])\
    .setOutputCol("cvx_code")\
    .setDistanceFunction("EUCLIDEAN")

result = light_model.fullAnnotate(["Sinovac", "Moderna", "BIOTHRAX"])
```

*Results:*

```bash
+----------+--------+-------------------------------------------------------+--------+
|ner_chunk |cvx_code|resolved_text                                          |Status  |
+----------+--------+-------------------------------------------------------+--------+
|Sinovac   |511     |COVID-19 IV Non-US Vaccine (CoronaVac, Sinovac)        |Non-US  |
|Moderna   |227     |COVID-19, mRNA, LNP-S, PF, pediatric 50 mcg/0.5 mL dose|Inactive|
|BIOTHRAX  |24      |anthrax                                                |Active  |
+----------+--------+-------------------------------------------------------+--------+
```

+ `cvx_resolver_pipeline`: This pretrained pipeline maps entities with their corresponding CVX codes.

*Example:*

```python
from sparknlp.pretrained import PretrainedPipeline

resolver_pipeline = PretrainedPipeline("cvx_resolver_pipeline", "en", "clinical/models")

text= "The patient has a history of influenza vaccine, tetanus and DTaP"
result = resolver_pipeline.fullAnnotate(text)
```

*Results:*

```bash
+-----------------+---------+--------+
|chunk            |ner_chunk|cvx_code|
+-----------------+---------+--------+
|influenza vaccine|Vaccine  |160     |
|tetanus          |Vaccine  |35      |
|DTaP             |Vaccine  |20      |
+-----------------+---------+--------+
```

</div><div class="prev_ver h3-box" markdown="1">

#### Updated Clinical NER Models With New Labels

`ner_jsl` and `ner_covid_trials` models were updated with the new label called "**Vaccine_Name**".

*Example:*

```python
...
jsl_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models") \
		.setInputCols(["sentence", "token", "embeddings"]) \
		.setOutputCol("jsl_ner")
...

sample_text= """The patient is a 21-day-old Caucasian male here for 2 days, there is no side effect observed after the influenza vaccine"""
```

*Results:*

```bash
|chunks            |   begin |   end | entities       |
|------------------|--------:|------:|:---------------|
|21-day-old        |      18 |    27 | Age            |
|Caucasian         |      29 |    37 | Race_Ethnicity |
|male              |      39 |    42 | Gender         |
|for 2 days        |      49 |    58 | Duration       |
|influenza vaccine |     100 |   116 | Vaccine_Name   |
```

</div><div class="prev_ver h3-box" markdown="1">

#### New Certification Training Notebooks for the `johnsnowlabs` Library

Now we have 46 new [Healtcare Certification Training notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings_JSL/Healthcare) for the users who want to use the new `johnsnowlabs` library.

</div><div class="prev_ver h3-box" markdown="1">

#### New and Updated Notebooks

+ New [Coreference Resolution](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/28.EntityRuler_with_Clinical_NER_Models.ipynb) notebook to find other references of clinical entities in a document.

+ Updated [Clinical Name Entity Recognition Model](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb) notebook with the new feature `setIgnoreStopWords` parameter and `ModelTracer` module.

+ Updated [Clinical Assertion Model](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/2.Clinical_Assertion_Model.ipynb) notebook with the new changes in `AssertionFilter` improvement.

+ Updated [Clinical Relation Extraction](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/10.Clinical_Relation_Extraction.ipynb) notebook with the new `setRelationDirectionCol` parameter in `RelationExtractionApproach`.

+ Updated [Date Normalizer](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/25.Date_Normalizer.ipynb) notebook with the new `setOutputDateformat` parameter in `DateNormalizer` and `Replacer` annotator.

+ Updated 25 Certification Training [Public notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Public) and 47 Certification Training [Healthcare notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare) with the latest updates in the libraries.

+ Updated 6 Databricks [Public notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/products/databricks/public) and 14 Databricks [Healthcare notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/products/databricks/healthcare) with the latest updates in the libraries and 4 new Databricks notebooks created.


</div><div class="prev_ver h3-box" markdown="1">

#### 6 New Clinical Models and Pipelines Added & Updated in Total

+ `cvx_code_mapper`
+ `cvx_name_mapper`
+ `sbiobertresolve_cvx`
+ `cvx_resolver_pipeline`
+ `ner_jsl`
+ `ner_covid_trials`


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_4_2_1">Version 4.2.1</a>
    </li>
    <li>
        <strong>Version 4.2.1</strong>
    </li>
</ul>
<ul class="pagination owl-carousel pagination_big">
    <li class="active"><a href="release_notes_4_2_1">4.2.1</a></li>
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
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