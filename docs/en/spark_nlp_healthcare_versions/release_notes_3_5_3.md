---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.5.3
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_5_3
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.5.3

#### Highlights

+ New `rxnorm_mapper` model
+ New `ChunkMapperFilterer` annotator to filter `ChunkMapperModel` results
+ New features
  - Add the `setReplaceLabels` parameter that allows replacing the non-conventional labels without using an external source file in the `NerConverterInternal()`.
  - Case sensitivity can be set in `ChunkMapperApproach` and `ChunkMapperModel` through `setLowerCase()` parameter.
  - Return multiple relations at a time in `ChunkMapperModel` models via `setRels()` parameter.
  - Filter the multi-token chunks separated with whitespace in `ChunkMapperApproach` by `setAllowMultiTokenChunk()` parameter.
+ New license validation policy in License Validator.
+ Bug fixes
+ Updated notebooks
+ List of recently updated or added models

#### New `rxnorm_mapper` Model

We are releasing `rxnorm_mapper` model that maps clinical entities and concepts to corresponding rxnorm codes.

See [Model Hub Page](https://nlp.johnsnowlabs.com/2022/06/07/rxnorm_mapper_en_3_0.html) for details.

*Example* :
```python
...
chunkerMapper = ChunkMapperModel.pretrained("rxnorm_mapper", "en", "clinical/models")\
       .setInputCols(["ner_chunk"])\
       .setOutputCol("mappings")\
       .setRel("rxnorm_code")
...

sample_text = "The patient was given Zyrtec 10 MG, Adapin 10 MG Oral Capsule, Septi-Soothe 0.5 Topical Spray"
```

*Results* :

```bash
 +------------------------------+---------------+
 |chunk                         |rxnorm_mappings|
 +------------------------------+---------------+
 |Zyrtec 10 MG                  |1011483        |
 |Adapin 10 MG Oral Capsule     |1000050        |
 |Septi-Soothe 0.5 Topical Spray|1000046        |
 +------------------------------+---------------+
```

#### New `ChunkMapperFilterer` Annotator to Filter `ChunkMapperModel` Results
`ChunkMapperFilterer` annotator allows filtering of the chunks that were passed through the `ChunkMapperModel`.
If `setReturnCriteria()` is set as `"success"`, only the chunks which are mapped by `ChunkMapperModel` are returned. Otherwise, if `setReturnCriteria()` is set as `"fail"`, only the chunks which are not mapped by ChunkMapperModel are returned.

*Example* :
```python
...
cfModel = ChunkMapperFilterer() \
            .setInputCols(["ner_chunk","mappings"]) \
            .setOutputCol("chunks_filtered")\
            .setReturnCriteria("success") #or "fail"
...
sample_text = "The patient was given Warfarina Lusa and amlodipine 10 mg. Also, he was given Aspagin, coumadin 5 mg and metformin"

```



`.setReturnCriteria("success")` *Results* :

```bash
+-----+---+--------------+--------------+
|begin|end|        entity|      mappings|
+-----+---+--------------+--------------+
|   22| 35|          DRUG|Warfarina Lusa|
+-----+---+--------------+--------------+
```

`.setReturnCriteria("fail")` *Results* :

```bash
+-----+---+--------+------------+
|begin|end|  entity|  not mapped|
+-----+---+--------+------------+
|   41| 50|    DRUG|  amlodipine|
|   80| 86|    DRUG|     Aspagin|
|   89| 96|    DRUG|    coumadin|
|  115|123|    DRUG|   metformin|
+-----+---+--------+------------+
```



#### New Features:

##### Add `setReplaceLabels` Parameter That Allows Replacing the Non-Conventional Labels Without Using an External Source File in the `NerConverterInternal()`.

Now you can replace the labels in NER models with custom labels by using `.setReplaceLabels` parameter with `NerConverterInternal` annotator. In this way, you will not need to use any other external source file to replace the labels with custom ones.

*Example* :

```python
...
clinical_ner = MedicalNerModel.pretrained("ner_jsl", "en", "clinical/models")\
    .setInputCols(["sentence","token", "word_embeddings"])\
    .setOutputCol("ner")

ner_converter_original = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("original_label")

ner_converter_replaced = NerConverterInternal()\
    .setInputCols(["sentence", "token", "ner"]) \
    .setOutputCol("replaced_label")\
    .setReplaceLabels({"Drug_Ingredient" : "Drug",'Drug_BrandName':'Drug'})
...

sample_text = "The patient was given Warfarina Lusa and amlodipine 10 mg. Also, he was given Aspagin, coumadin 5 mg, and metformin"
```

*Results* :

```bash
+--------------+-----+---+---------------+--------------+
|chunk         |begin|end|original_label |replaced_label|
+--------------+-----+---+---------------+--------------+
|Warfarina Lusa|22   |35 |Drug_BrandName |Drug          |
|amlodipine    |41   |50 |Drug_Ingredient|Drug          |
|10 mg         |52   |56 |Strength       |Strength      |
|he            |65   |66 |Gender         |Gender        |
|Aspagin       |78   |84 |Drug_BrandName |Drug          |
|coumadin      |87   |94 |Drug_Ingredient|Drug          |
|5 mg          |96   |99 |Strength       |Strength      |
|metformin     |106  |114|Drug_Ingredient|Drug          |
+--------------+-----+---+---------------+--------------+
```


##### Case Sensitivity in `ChunkMapperApproach` and `ChunkMapperModel` Through `setLowerCase()` Parameter

The case status of `ChunkMapperApproach` and `ChunkMapperModel` can be set by using `setLowerCase()` parameter.

*Example* :

```python
...
chunkerMapperapproach = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("mappings.json") \
        .setRel("action") \
        .setLowerCase(True) #or False

...

sentences = [["""The patient was given Warfarina lusa and amlodipine 10 mg, coumadin 5 mg.
                 The patient was given Coumadin"""]]
```

`setLowerCase(True)` *Results* :

```bash
+------------------------+-----------+
|chunk                   |mapped     |
+------------------------+-----------+
|Warfarina lusa          |540228     |
|amlodipine              |329526     |
|coumadin                |202421     |
|Coumadin                |202421     |
+------------------------+-----------+

```

`setLowerCase(False)` *Results* :

```bash
+------------------------+-----------+
|chunk                   |mapped     |
+------------------------+-----------+
|Warfarina lusa          |NONE       |
|amlodipine              |329526     |
|coumadin                |NONE       |
|Coumadin                |202421     |
+------------------------+-----------+
```

##### Return Multiple Relations At a Time In ChunkMapper Models Via `setRels()` Parameter
Multiple relations for the same chunk can be set with the `setRels()` parameter in both `ChunkMapperApproach` and `ChunkMapperModel`.

*Example* :
```python
...
chunkerMapperapproach = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("mappings.json") \
        .setRels(["action","treatment"]) \
        .setLowerCase(True) \
...

sample_text = "The patient was given Warfarina Lusa."
```

*Results* :

```bash
+-----+---+--------------+-------------+---------+
|begin|end|        entity|     mappings| relation|
+-----+---+--------------+-------------+---------+
|   22| 35|Warfarina Lusa|Anticoagulant|   action|
|   22| 35|Warfarina Lusa|Heart Disease|treatment|
+-----+---+--------------+-------------+---------+
```

##### Filter the Multi-Token Chunks Separated With Whitespace in `ChunkMapperApproach` and `ChunkMapperModel` by `setAllowMultiTokenChunk()` Parameter

The chunks that include multi-tokens separated by a whitespace, can be filtered by using `setAllowMultiTokenChunk()` parameter.

*Example* :
```python
...
chunkerMapper = ChunkMapperApproach() \
        .setInputCols(["ner_chunk"]) \
        .setOutputCol("mappings") \
        .setDictionary("mappings.json") \
        .setLowerCase(True) \
        .setRels(["action", "treatment"]) \
        .setAllowMultiTokenChunk(False)
...

sample_text = "The patient was given Warfarina Lusa"
```

`setAllowMultiTokenChunk(False)` *Results* :

```bash
+-----+---+--------------+--------+--------+
|begin|end|         chunk|mappings|relation|
+-----+---+--------------+--------+--------+
|   22| 35|Warfarina Lusa|    NONE|    null|
+-----+---+--------------+--------+--------+
```

`setAllowMultiTokenChunk(True)` *Results* :

```bash
+-----+---+--------------+-------------+---------+
|begin|end|         chunk|     mappings| relation|
+-----+---+--------------+-------------+---------+
|   22| 35|Warfarina Lusa|Anticoagulant|   action|
|   22| 35|Warfarina Lusa|Heart Disease|treatment|
+-----+---+--------------+-------------+---------+
```



#### New License Validation Policies in License Validator

A new version of the License Validator has been included in Spark NLP for Healthcare. This License Validator checks the compatibility between the type of your license and the environment you are using, allowing the license to be used only for the environment it was requested (single-node, cluster, databricks, etc) and the number of concurrent sessions (floating or not-floating). You can check which type of license you have in [my.johnsnowlabs.com](https://my.johnsnowlabs.com/) -> My Subscriptions.

If your license stopped working, please contact [support@johnsnowlabs.com](mailto:support@johnsnowlabs.com) so that it can be checked the difference between the environment your license was requested for and the one it's currently being used.

#### Bug Fixes

We fixed some issues in `AnnotationToolJsonReader` tool, `DrugNormalizer` and `ContextualParserApproach` annotators.

+ **`DrugNormalizer`** : Fixed some issues that affect the performance.
+ **`ContextualParserApproach`** : Fixed the issue in the computation of indices for documents with more than one sentence while defining the rule-scope field as a document.
+ **`AnnotationToolJsonReader`** : Fixed an issue where relation labels were not being extracted from the Annotation Lab json file export.

#### Updated Notebooks

+ [Clinical Named Entity Recognition Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb) <br/>
 `.setReplaceLabels` parameter example was added.
+ [Chunk Mapping Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb) <br/>
 New case sensitivity, selecting multiple relations, filtering multi-token chunks and `ChunkMapperFilterer` features were added.

#### List of Recently Updated Models

- `sbiobertresolve_icdo_augmented`
- `rxnorm_mapper`

For all Spark NLP for healthcare models, please check: [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Spark+NLP+for+Healthcare)

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_5_2">Version 3.5.2</a>
    </li>
    <li>
        <strong>Version 3.5.3</strong>
    </li>
    <li>
        <a href="release_notes_4_0_0">Version 4.0.0</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li class="active"><a href="release_notes_3_5_3">3.5.3</a></li>
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