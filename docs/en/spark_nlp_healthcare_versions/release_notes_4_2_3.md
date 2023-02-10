---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 4.2.3
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_4_2_3
key: docs-licensed-release-notes
modify_date: 2022-12-02
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

## 4.2.3

#### Highlights

+ 3 new chunk mapper models to mapping Drugs and Diseases from the KEGG Database as well as mapping abbreviations to their categories
+ New utility & helper Relation Extraction modules to handle preprocess
+ New utility & helper OCR modules to handle annotate
+ New utility & helper NER log parser
+ Adding flexibility chunk merger prioritization
+ Core improvements and bug fixes
+ New and updated notebooks
+ 3 new clinical models and pipelines added & updated in total


</div><div class="prev_ver h3-box" markdown="1">


#### 3 New Hhunk Mapper Models to Mapping Drugs and Diseases from the KEGG Database as well as Mapping Abbreviations to Their Categories

+ `kegg_disease_mapper`: This pretrained model maps diseases with their corresponding `category`, `description`, `icd10_code`, `icd11_code`, `mesh_code`, and hierarchical `brite_code`. This model was trained with the data from the KEGG database.

*Example:*

```python
chunkerMapper = ChunkMapperModel.pretrained("kegg_disease_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["description", "category", "icd10_code", "icd11_code", "mesh_code", "brite_code"])

text= "A 55-year-old female with a history of myopia, kniest dysplasia and prostate cancer. She was on glipizide , and dapagliflozin for congenital nephrogenic diabetes insipidus."
```

*Result:*

```bash
+-----------------------------------------+--------------------------------------------------+-----------------------+----------+----------+---------+-----------------------+
|                                ner_chunk|                                       description|               category|icd10_code|icd11_code|mesh_code|             brite_code|
+-----------------------------------------+--------------------------------------------------+-----------------------+----------+----------+---------+-----------------------+
|                                   myopia|Myopia is the most common ocular disorder world...| Nervous system disease|     H52.1|    9D00.0|  D009216|            08402,08403|
|                         kniest dysplasia|Kniest dysplasia is an autosomal dominant chond...|Congenital malformation|     Q77.7|    LD24.3|  C537207|            08402,08403|
|                          prostate cancer|Prostate cancer constitutes a major health prob...|                 Cancer|       C61|      2C82|     NONE|08402,08403,08442,08441|
|congenital nephrogenic diabetes insipidus|Nephrogenic diabetes insipidus (NDI) is charact...| Urinary system disease|     N25.1|   GB90.4A|  D018500|            08402,08403|
+-----------------------------------------+--------------------------------------------------+-----------------------+----------+----------+---------+-----------------------+
```


+ `kegg_drug_mapper`: This pretrained model maps drugs with their corresponding `efficacy`, `molecular_weight` as well as `CAS`, `PubChem`, `ChEBI`, `LigandBox`, `NIKKAJI`, `PDB-CCD` codes. This model was trained with the data from the KEGG database.

*Example*:

```python
chunkerMapper = ChunkMapperModel.pretrained("kegg_drug_mapper", "en", "clinical/models")\
    .setInputCols(["ner_chunk"])\
    .setOutputCol("mappings")\
    .setRels(["efficacy", "molecular_weight", "CAS", "PubChem", "ChEBI", "LigandBox", "NIKKAJI", "PDB-CCD"])

text= "She is given OxyContin, folic acid, levothyroxine, Norvasc, aspirin, Neurontin"
```

*Result*:

```bash
+-------------+--------------------------------------------------+----------------+----------+-----------+-------+---------+---------+-------+
|    ner_chunk|                                          efficacy|molecular_weight|       CAS|    PubChem|  ChEBI|LigandBox|  NIKKAJI|PDB-CCD|
+-------------+--------------------------------------------------+----------------+----------+-----------+-------+---------+---------+-------+
|    OxyContin|     Analgesic (narcotic), Opioid receptor agonist|        351.8246|  124-90-3|  7847912.0| 7859.0|   D00847|J281.239H|   NONE|
|   folic acid|Anti-anemic, Hematopoietic, Supplement (folic a...|        441.3975|   59-30-3|  7847138.0|27470.0|   D00070|  J1.392G|    FOL|
|levothyroxine|                     Replenisher (thyroid hormone)|          776.87|   51-48-9|9.6024815E7|18332.0|   D08125|  J4.118A|    T44|
|      Norvasc|Antihypertensive, Vasodilator, Calcium channel ...|        408.8759|88150-42-9|5.1091781E7| 2668.0|   D07450| J33.383B|   NONE|
|      aspirin|Analgesic, Anti-inflammatory, Antipyretic, Anti...|        180.1574|   50-78-2|  7847177.0|15365.0|   D00109|  J2.300K|    AIN|
|    Neurontin|                     Anticonvulsant, Antiepileptic|        171.2368|60142-96-3|  7847398.0|42797.0|   D00332| J39.388F|    GBN|
+-------------+--------------------------------------------------+----------------+----------+-----------+-------+---------+---------+-------+
```

+ `abbreviation_category_mapper`: This pretrained model maps abbreviations and acronyms of medical regulatory activities with their definitions and categories.
Predicted categories: `general`, `problem`, `test`, `treatment`, `medical_condition`, `clinical_dept`, `drug`, `nursing`, `internal_organ_or_component`, `hospital_unit`, `drug_frequency`, `employment`, `procedure`.

*Example*:
```python
chunkerMapper = ChunkMapperModel.pretrained("abbreviation_category_mapper", "en", "clinical/models")\
     .setInputCols(["abbr_ner_chunk"])\
     .setOutputCol("mappings")\
     .setRels(["definition", "category"])\

text = ["""Gravid with estimated fetal weight of 6-6/12 pounds.
         LABORATORY DATA: Laboratory tests include a CBC which is normal.
         VDRL: Nonreactive
         HIV: Negative. One-Hour Glucose: 117. Group B strep has not been done as yet."""]

```

*Result*:

```bash
| chunk   | category          | definition                             |
|:--------|:------------------|:---------------------------------------|
| CBC     | general           | complete blood count                   |
| VDRL    | clinical_dept     | Venereal Disease Research Laboratories |
| HIV     | medical_condition | Human immunodeficiency virus           |
```

</div><div class="prev_ver h3-box" markdown="1">

####  New Utility & Helper Relation Extraction Modules to Handle Preprocess

This process is standard and training column should be same in all RE trainings. We can simplify this process with helper class. With proposed changes it can be done as follows:

*Example*:

```python
from sparknlp_jsl.training import REDatasetHelper

# map entity columns to dataset columns
column_map = {
    "begin1": "firstCharEnt1",
    "end1": "lastCharEnt1",
    "begin2": "firstCharEnt2",
    "end2": "lastCharEnt2",
    "chunk1": "chunk1",
    "chunk2": "chunk2",
    "label1": "label1",
    "label2": "label2"
}

# apply preprocess function to dataframe
data = REDatasetHelper(data).create_annotation_column(
    column_map,
    ner_column_name="train_ner_chunks" # optional, default train_ner_chunks
)
```


</div><div class="prev_ver h3-box" markdown="1">

####  New Utility & Helper OCR Modules to Handle Annotations

This modeule can generates an annotated PDF file using input PDF files. `style`:  PDF file proccess style that has 3 options;
- `black_band`: Black bands over the chunks detected by NER pipeline.
- `bounding_box`: Colorful bounding boxes around the chunks detected by NER pipeline. Each color represents a different NER label.
- `highlight`: Colorful highlights over the chunks detected by NER pipeline. Each color represents a different NER label.
- 
You can check [Spark OCR Utility Module](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/5.3.Spark_OCR_Utility_Module.ipynb) notebook for more examples.

*Example*:

```python
from sparknlp_jsl.utils.ocr_nlp_processor import  ocr_entity_processor

path='/*.pdf'
        
box = "bounding_box"
ocr_entity_processor(spark=spark,file_path=path,ner_pipeline = nlp_model,chunk_col = "merged_chunk", black_list = ["AGE", "DATA", "PATIENT"],
                    style = box, save_dir = "colored_box",label= True, label_color = "red",color_chart_path = "label_colors.png", display_result=True)

box = "highlight"
ocr_entity_processor(spark=spark,file_path=path, ner_pipeline = nlp_model, chunk_col = "merged_chunk", black_list = ["AGE", "DATE", "PATIENT"],
                    style = box, save_dir = "colored_box", label= True, label_color = "red", color_chart_path = "label_colors.png", display_result=True)

box = "black_band"
ocr_entity_processor(spark=spark,file_path=path, ner_pipeline = nlp_modelchunk_col = "merged_chunk", 
                     style = box, save_dir = "black_band",label= True, label_color = "red", display_result = True)
```

*Results*:

- **Bounding box with labels and black list**

![image](https://user-images.githubusercontent.com/64752006/205349180-8222ec38-8b43-44a2-abf2-def72fd82d68.png)


- **Highlight with labels and black_list**

![image](https://user-images.githubusercontent.com/64752006/205343977-7f54eb4e-4a3c-4be2-8acf-2f0f3b7dd0f5.png)

- **black_band with labels**

![image](https://user-images.githubusercontent.com/64752006/205343389-9563a9ce-f971-4865-b94f-4b0cc92e2bcc.png)





</div><div class="prev_ver h3-box" markdown="1">


#### New Utility & Helper NER Log Parser

`ner_utils`: This new module is used after NER training to calculate mertic chunkbase and plot training logs.

*Example*:
```python
nerTagger = NerDLApproach()\
              .setInputCols(["sentence", "token", "embeddings"])\
              .setLabelColumn("label")\
              .setOutputCol("ner")\
              ...  
              .setOutputLogsPath('ner_logs')
    
ner_pipeline = Pipeline(stages=[glove_embeddings,
                                graph_builder,
                                nerTagger])

ner_model = ner_pipeline.fit(training_data)

```

- `evaluate`: if verbose, returns overall performance, as well as performance per chunk type; otherwise, simply returns overall precision, recall, f1 scores

*Example*:

```python
from sparknlp_jsl.utils.ner_utils import evaluate

metrics = evaluate(preds_df['ground_truth'].values, preds_df['prediction'].values)

```

*Result:*

```bash
processed 14133 tokens with 1758 phrases; found: 1779 phrases; correct: 1475.
accuracy:  83.45%; (non-O)
accuracy:  96.67%; precision:  82.91%; recall:  83.90%; FB1:  83.40
              LOC: precision:  91.41%; recall:  85.69%; FB1:  88.46  524
             MISC: precision:  78.15%; recall:  62.11%; FB1:  69.21  151
              ORG: precision:  61.86%; recall:  74.93%; FB1:  67.77  430
              PER: precision:  90.80%; recall:  93.58%; FB1:  92.17  674
```


- `loss_plot`: Plots the figure of loss vs epochs

*Example*:

```python
from sparknlp_jsl.utils.ner_utils import loss_plot

loss_plot('./ner_logs/'+log_files[0])
```

*Results*:

![image](https://user-images.githubusercontent.com/64752006/205368367-f70bb792-b1ff-41cd-8ed5-8dee94f10aa8.png)


- `get_charts` : Plots the figures of metrics ( precision, recall, f1) vs epochs

*Example*:

```python
from sparknlp_jsl.utils.ner_utils import get_charts

get_charts('./ner_logs/'+log_files[0])
```

*Results*:

![image](https://user-images.githubusercontent.com/64752006/205368210-f5ffb64c-8a22-4758-8423-ade6ac3ee8cd.png)

![image](https://user-images.githubusercontent.com/64752006/205368132-c86e7c2b-555e-4653-a799-c55cc5e9e9a0.png)



</div><div class="prev_ver h3-box" markdown="1">

#### Adding Flexibility Chunk Merger Prioritization

`orderingFeatures`: Array of strings specifying the ordering features to use for overlapping entities. Possible values are ChunkBegin, ChunkLength, ChunkPrecedence, ChunkConfidence

`selectionStrategy`: Whether to select annotations sequentially based on annotation order `Sequential` or using any other available strategy, currently only `DiverseLonger` are available.

`defaultConfidence`: When ChunkConfidence ordering feature is included and a given annotation does not have any confidence the value of this param will be used.

`chunkPrecedence`: When ChunkPrecedence ordering feature is used this param contains the comma separated fields in metadata that drive prioritization of overlapping annotations. When used by itself (empty chunkPrecedenceValuePrioritization) annotations will be prioritized based on number of metadata fields present. When used together with chunkPrecedenceValuePrioritization param it will prioritize based on the order of its values.

`chunkPrecedenceValuePrioritization`: When ChunkPrecedence ordering feature is used this param contains an Array of comma separated values representing the desired order of prioritization for the VALUES in the metadata fields included from chunkPrecedence.

*Example*:

```python
text = """A 63 years old man presents to the hospital with a history of recurrent infections 
that include cellulitis, pneumonias, and upper respiratory tract infections..."""

+-------------------------------------------------------------------------------------+
|ner_deid_chunk                                                                       |
+-------------------------------------------------------------------------------------+
|[{chunk, 2, 3, 63, {entity -> AGE, sentence -> 0, chunk -> 0, confidence -> 0.9997}}]|
+-------------------------------------------------------------------------------------+

+----------------------------------------------------------------------------------------------------+
|jsl_ner_chunk                                                                                       |     
+----------------------------------------------------------------------------------------------------+
|[{chunk, 2, 13, 63 years old, {entity -> Age, sentence -> 0, chunk -> 0, confidence -> 0.85873336}}]|
+----------------------------------------------------------------------------------------------------+

```

- **Merging overlapped chunks by considering their lenght** <br/>
If we set `setOrderingFeatures(["ChunkLength"])` and `setSelectionStrategy("DiverseLonger")` parameters, the longest chunk will be prioritized in case of overlapping. 

*Example*:

```python
chunk_merger = ChunkMergeApproach()\
    .setInputCols('ner_deid_chunk', "jsl_ner_chunk")\
    .setOutputCol('merged_ner_chunk')\
    .setOrderingFeatures(["ChunkLength"])\
    .setSelectionStrategy("DiverseLonger")
```

*Results*:

```bash
|begin|end|        chunk|         entity|
+-----+---+-------------+---------------+
|    2| 13| 63 years old|            Age|
|   15| 17|          man|         Gender|
|   35| 42|     hospital|  Clinical_Dept|

```

- **Merging overlapped chunks by considering custom values that we set** <br/>
`setChunkPrecedence()` parameter contains an Array of comma separated values representing the desired order of prioritization for the VALUES in the metadata fields included from `setOrderingFeatures(["chunkPrecedence"])`.

*Example*:

```python
chunk_merger = ChunkMergeApproach()\
    .setInputCols('ner_deid_chunk', "jsl_ner_chunk")\
    .setOutputCol('merged_ner_chunk')\
    .setMergeOverlapping(True) \
    .setOrderingFeatures(["ChunkPrecedence"]) \
    .setChunkPrecedence('ner_deid_chunk,AGE') \
#    .setChunkPrecedenceValuePrioritization(["ner_deid_chunk,AGE", "jsl_ner_chunk,Age"]) 
```

*Results*:

```bash
|begin|end|        chunk|         entity|
+-----+---+-------------+---------------+
|    2|  3|           63|            AGE|
|   15| 17|          man|         Gender|
|   35| 42|     hospital|  Clinical_Dept|

```

You can check [NER Chunk Merger](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/7.Clinical_NER_Chunk_Merger.ipynb) notebook for more examples.




</div><div class="prev_ver h3-box" markdown="1">

#### Core improvements and bug fixes

- AssertionDL IncludeConfidence() parameters default value set by True
- Fixed NaN outputs in RelationExtraction
- Fixed loadSavedModel method that we use for importing transformers into Spark NLP
- Fixed replacer with setUseReplacement(True) parameter
- Added overall confidence score to MedicalNerModel when setIncludeAllConfidenceScore is True
- Fixed in InternalResourceDownloader showAvailableAnnotators

</div><div class="prev_ver h3-box" markdown="1">


#### New and Updated Notebooks

+ New [Spark OCR Utility Module](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/5.3.Spark_OCR_Utility_Module.ipynb) notebook to help handle OCR process.

+ Updated [Clinical Entity Resolvers](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb) notebook with `Assertion Filterer` example.

+ Updated [NER Chunk Merger](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/7.Clinical_NER_Chunk_Merger.ipynb) notebook with flexibility chunk merger prioritization example.

+ Updated [Clinical Relation Extraction](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb) notebook with new `REDatasetHelper` module.

+ Updated [ALab Module SparkNLP JSL](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Annotation_Lab/Complete_ALab_Module_SparkNLP_JSL.ipynb) notebook with new updates.


</div><div class="prev_ver h3-box" markdown="1">

#### 3 New Clinical Models and Pipelines Added & Updated in Total

+ `kegg_disease_mapper`
+ `kegg_drug_mapper`
+ `abbreviation_category_mapper`

For all Spark NLP for healthcare models, please check: [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Healthcare+NLP)


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-healthcare-pagination.html -%}