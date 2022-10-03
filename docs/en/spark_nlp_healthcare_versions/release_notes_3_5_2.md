---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.5.2
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_5_2
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.5.2

#### Highlights

+ `TFGraphBuilder` annotator to create graphs for training NER, Assertion, Relation Extraction, and Generic Classifier models
+ Default TF graphs added for `AssertionDLApproach` to let users train models without custom graphs
+ New functionalities in `ContextualParserApproach`
+ Printing the list of clinical pretrained models and pipelines with one-liner
+ New clinical models
  - Clinical NER model (`ner_biomedical_bc2gm`)
  - Clinical `ChunkMapper` models (`abbreviation_mapper`, `rxnorm_ndc_mapper`, `drug_brandname_ndc_mapper`, `rxnorm_action_treatment_mapper`)
+ Bug fixes
+ New and updated notebooks
+ List of recently updated or added models

#### `TFGraphBuilder` annotator to create graphs for Training NER, Assertion, Relation Extraction, and Generic Classifier Models

We have a new annotator used to create graphs in the model training pipeline. `TFGraphBuilder` inspects the data and creates the proper graph if a suitable version of TensorFlow (<= 2.7 ) is available. The graph is stored in the defined folder and loaded by the approach.

You can use this builder with `MedicalNerApproach`, `RelationExtractionApproach`, `AssertionDLApproach`, and `GenericClassifierApproach`

*Example:*

```python
graph_folder_path = "./medical_graphs"

med_ner_graph_builder = TFGraphBuilder()\
    .setModelName("ner_dl")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setLabelColumn("label")\
    .setGraphFile("auto")\
    .setHiddenUnitsNumber(20)\
    .setGraphFolder(graph_folder_path)

med_ner = MedicalNerApproach() \
    ...
    .setGraphFolder(graph_folder)

medner_pipeline = Pipeline()([
    ...,
    med_ner_graph_builder,
    med_ner    
    ])
```

For more examples, please check [TFGraph Builder Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/17.Graph_builder_for_DL_models.ipynb).

#### Default TF graphs added for `AssertionDLApproach` to let users train models without custom graphs

We added default TF graphs for the `AssertionDLApproach` to let users train assertion models without specifying any custom TF graph.

**Default Graph Features:**
+ Feature Sizes: 100, 200, 768
+ Number of Classes: 2, 4, 8

#### New Functionalities in `ContextualParserApproach`

+ Added `.setOptionalContextRules` parameter that allows to output regex matches regardless of context match (prefix, suffix configuration).
+ Allows sending a JSON string of the configuration file to `setJsonPath` parameter.

**Confidence Value Scenarios:**

1. When there is regex match only, the confidence value will be 0.5.
2. When there are regex and prefix matches together, the confidence value will be > 0.5 depending on the distance between target token and the prefix.
3. When there are regex and suffix matches together, the confidence value will be > 0.5 depending on the distance between target token and the suffix.
4. When there are regex, prefix, and suffix matches all together, the confidence value will be > than the other scenarios.

*Example*:

```python
jsonString = {
    "entity": "CarId",
    "ruleScope": "sentence",
    "completeMatchRegex": "false",
    "regex": "\\d+",
    "prefix": ["red"],
    "contextLength": 100
}

with open("jsonString.json", "w") as f:
    json.dump(jsonString, f)

contextual_parser = ContextualParserApproach()\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("entity")\
    .setJsonPath("jsonString.json")\
    .setCaseSensitive(True)\
    .setOptionalContextRules(True)
```

#### Printing the List of Clinical Pretrained Models and Pipelines with One-Liner

  Now we can check what the clinical model names are of a specific annotator and the names of clinical pretrained pipelines in a language.

  + **Listing Clinical Model Names:**

*Example*:

```python
from sparknlp_jsl.pretrained import InternalResourceDownloader

InternalResourceDownloader.showPrivateModels("AssertionDLModel")
```

*Results*:

```bash
+-----------------------------------+------+---------+
| Model                             | lang | version |
+-----------------------------------+------+---------+
| assertion_ml                      |  en  | 2.0.2   |
| assertion_dl                      |  en  | 2.0.2   |
| assertion_dl_healthcare           |  en  | 2.7.2   |
| assertion_dl_biobert              |  en  | 2.7.2   |
| assertion_dl                      |  en  | 2.7.2   |
| assertion_dl_radiology            |  en  | 2.7.4   |
| assertion_jsl_large               |  en  | 3.1.2   |
| assertion_jsl                     |  en  | 3.1.2   |
| assertion_dl_scope_L10R10         |  en  | 3.4.2   |
| assertion_dl_biobert_scope_L10R10 |  en  | 3.4.2   |
+-----------------------------------+------+---------+
```

+ **Listing Clinical Pretrained Pipelines:**

```python
from sparknlp_jsl.pretrained import InternalResourceDownloader

InternalResourceDownloader.showPrivatePipelines("en")
```

```bash
+--------------------------------------------------------+------+---------+
| Pipeline                                               | lang | version |
+--------------------------------------------------------+------+---------+
| clinical_analysis                                      |  en  | 2.4.0   |
| clinical_ner_assertion                                 |  en  | 2.4.0   |
| clinical_deidentification                              |  en  | 2.4.0   |
| clinical_analysis                                      |  en  | 2.4.0   |
| explain_clinical_doc_ade                               |  en  | 2.7.3   |
| icd10cm_snomed_mapping                                 |  en  | 2.7.5   |
| recognize_entities_posology                            |  en  | 3.0.0   |
| explain_clinical_doc_carp                              |  en  | 3.0.0   |
| recognize_entities_posology                            |  en  | 3.0.0   |
| explain_clinical_doc_ade                               |  en  | 3.0.0   |
| explain_clinical_doc_era                               |  en  | 3.0.0   |
| icd10cm_snomed_mapping                                 |  en  | 3.0.2   |
| snomed_icd10cm_mapping                                 |  en  | 3.0.2   |
| icd10cm_umls_mapping                                   |  en  | 3.0.2   |
| snomed_umls_mapping                                    |  en  | 3.0.2   |
| ...                                                    |  ... | ...     |
+--------------------------------------------------------+------+---------+
```

#### New `ner_biomedical_bc2gm` NER Model

This model has been trained to extract genes/proteins from a medical text.

See [Model Card](https://nlp.johnsnowlabs.com/2022/05/10/ner_biomedical_bc2gm_en_3_0.html) for more details.

*Example* :

```python
...
ner = MedicalNerModel.pretrained("ner_biomedical_bc2gm", "en", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"]) \
    .setOutputCol("ner")
...

text = spark.createDataFrame([["Immunohistochemical staining was positive for S-100 in all 9 cases stained, positive for HMB-45 in 9 (90%) of 10, and negative for cytokeratin in all 9 cases in which myxoid melanoma remained in the block after previous sections."]]).toDF("text")

result = model.transform(text)
```
*Results* :

```
+-----------+------------+
|chunk      |ner_label   |
+-----------+------------+
|S-100      |GENE_PROTEIN|
|HMB-45     |GENE_PROTEIN|
|cytokeratin|GENE_PROTEIN|
+-----------+------------+
```

#### New Clinical `ChunkMapper` Models

We have 4 new `ChunkMapper` models and a new [Chunk Mapping Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb) for showing their examples.

+ `drug_brandname_ndc_mapper`: This model maps drug brand names to corresponding National Drug Codes (NDC). Product NDCs for each strength are returned in result and metadata.

See [Model Card](https://nlp.johnsnowlabs.com/2022/05/11/drug_brandname_ndc_mapper_en_3_0.html) for more details.

*Example* :

```python
document_assembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("chunk")

chunkerMapper = ChunkMapperModel.pretrained("drug_brandname_ndc_mapper", "en", "clinical/models")\
      .setInputCols(["chunk"])\
      .setOutputCol("ndc")\
      .setRel("Strength_NDC")

model = PipelineModel(stages=[document_assembler,
                                 chunkerMapper])  

light_model = LightPipeline(model)
res = light_model.fullAnnotate(["zytiga", "ZYVOX", "ZYTIGA"])
```

*Results* :
```bash
+-------------+--------------------------+-----------------------------------------------------------+
| Brandname   | Strenth_NDC              | Other_NDSs                                                |
+-------------+--------------------------+-----------------------------------------------------------+
| zytiga      | 500 mg/1 | 57894-195     | ['250 mg/1 | 57894-150']                                  |
| ZYVOX       | 600 mg/300mL | 0009-4992 | ['600 mg/300mL | 66298-7807', '600 mg/300mL | 0009-7807'] |
| ZYTIGA      | 500 mg/1 | 57894-195     | ['250 mg/1 | 57894-150']                                  |
+-------------+--------------------------+-----------------------------------------------------------+

```

+ `abbreviation_mapper`: This model maps abbreviations and acronyms of medical regulatory activities with their definitions.

See [Model Card](https://nlp.johnsnowlabs.com/2022/05/11/abbreviation_mapper_en_3_0.html) for details.

*Example:*

```bash
input = ["""Gravid with estimated fetal weight of 6-6/12 pounds.
            LABORATORY DATA: Laboratory tests include a CBC which is normal. 
            HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet."""]
           
>> output:
+------------+----------------------------+
|Abbreviation|Definition                  |
+------------+----------------------------+
|CBC         |complete blood count        |
|HIV         |human immunodeficiency virus|
+------------+----------------------------+
```

+ `rxnorm_action_treatment_mapper`: RxNorm and RxNorm Extension codes with their corresponding action and treatment. Action refers to the function of the drug in various body systems; treatment refers to which disease the drug is used to treat.

See [Model Card](https://nlp.johnsnowlabs.com/2022/05/08/rxnorm_action_treatment_mapper_en_3_0.html) for more details.

*Example:*

```bash
input = ['Sinequan 150 MG', 'Zonalon 50 mg']
           
>> output:
+---------------+------------+---------------+
|chunk          |rxnorm_code |Action         |
+---------------+------------+---------------+
|Sinequan 150 MG|1000067     |Antidepressant |
|Zonalon 50 mg  |103971      |Analgesic      |
+---------------+------------+---------------+
```

+ `rxnorm_ndc_mapper`: This pretrained model maps RxNorm and RxNorm Extension codes with corresponding National Drug Codes (NDC).

See [Model Card](https://nlp.johnsnowlabs.com/2022/05/09/rxnorm_ndc_mapper_en_3_0.html) for more details.

*Example:*

```bash
input = ['doxepin hydrochloride 50 MG/ML', 'macadamia nut 100 MG/ML']
           
>> output:
+------------------------------+------------+------------+
|chunk                         |rxnorm_code |Product NDC |
+------------------------------+------------+------------+
|doxepin hydrochloride 50 MG/ML|1000091     |00378-8117  |
|macadamia nut 100 MG/ML       |212433      |00064-2120  |
+------------------------------+------------+------------+
```

#### Bug Fixes

We fixed some issues in `DrugNormalizer`, `DateNormalizer` and `ContextualParserApproach` annotators.

+ **`DateNormalizer`** : We fixed some relative date issues and also `DateNormalizer` takes account the Leap years now.
+ **`DrugNormalizer`** : Fixed some formats.
+ **`ContextualParserApproach`** :
  - Computing the right distance for prefix.
  - Extracting the right content for suffix.
  - Handling special characters in prefix and suffix.


#### New and Updated Notebooks
  - We prepared [Spark NLP for Healthcare 3hr Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/1hr_workshop/SparkNLP_for_Healthcare_3h_Notebook.ipynb) to cover mostly used components of Spark NLP in ODSC East 2022-3 hours hands-on workshop on 'Modular Approach to Solve Problems at Scale in Healthcare NLP'. You can also find its Databricks version [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/databricks/python/healthcare_tutorials/SparkNLP_for_Healthcare_3h_Notebook.ipynb).
  - New [Chunk Mapping Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/26.Chunk_Mapping.ipynb) for showing the examples of Chunk Mapper models.
  - [Updated healthcare tutorial notebooks for Databricks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare/databricks_notebooks) with `sparknlp_jsl` v3.5.1
  - We have a new [Databricks healthcare tutorials folder](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/databricks/python/healthcare_tutorials) in which you can find all Spark NLP for Healthcare Databricks tutorial notebooks. 
  - [Updated Graph Builder Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/17.Graph_builder_for_DL_models.ipynb) by adding the examples of new `TFGraphBuilder` annotator.

#### List of Recently Updated or Added Models

- `sbiobertresolve_rxnorm_action_treatment`
- `ner_biomedical_bc2gm`
- `abbreviation_mapper`
- `rxnorm_ndc_mapper`
- `drug_brandname_ndc_mapper`
- `sbiobertresolve_cpt_procedures_measurements_augmented`
- `sbiobertresolve_icd10cm_slim_billable_hcc`
- `sbiobertresolve_icd10cm_slim_normalized`

**For all Spark NLP for healthcare models, please check : [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Spark+NLP+for+Healthcare)**

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_5_1">Version 3.5.1</a>
    </li>
    <li>
        <strong>Version 3.5.2</strong>
    </li>
    <li>
        <a href="release_notes_3_5_3">Version 3.5.3</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li class="active"><a href="release_notes_3_5_2">3.5.2</a></li>
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