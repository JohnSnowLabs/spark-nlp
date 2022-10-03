---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.7.3
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_7_3
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

### 2.7.3

We are glad to announce that Spark NLP for Healthcare 2.7.3 has been released!  

#### Highlights:

- Introducing a brand-new **RelationExtractionDL Annotator** – Achieving SOTA results in clinical relation extraction using **BioBert**.
- Massive Improvements & feature enhancements in **De-Identification** module:
    - Introduction of **faker** augmentation in Spark NLP for Healthcare to generate random data for obfuscation in de-identification module.
    - Brand-new annotator for **Structured De-Identification**.
- **Drug Normalizer:**  Normalize medication-related phrases (dosage, form and strength) and abbreviations in text and named entities extracted by NER models.
- **Confidence scores** in **assertion** output : just like NER output, assertion models now also support confidence scores for each prediction.
- **Cosine similarity** metrics in entity resolvers to get more informative and semantically correct results.
- **AuxLabel** in the metadata of entity resolvers to return additional mappings.
- New **Relation Extraction** models to extract relations between **body parts** and clinical entities.
- New **Entity Resolver** models to extract billable medical codes.
- New **Clinical Pretrained NER** models.
- Bug fixes & general improvements.
- Matching the version with Spark NLP open-source v2.7.3.


#### 1. Improvements in De-Identification Module:

Integration of `faker` library to automatically generate random data like names, dates, addresses etc so users dont have to specify dummy data (custom obfuscation files can still be used). It also improves the obfuscation results due to a bigger pool of random values.

**How to use:**

Set the flag `setObfuscateRefSource` to `faker`

    deidentification = DeIdentification()
        .setInputCols(["sentence", "token", "ner_chunk"])\
		.setOutputCol("deidentified")\
		.setMode("obfuscate") \
		.setObfuscateRefSource("faker")

For more details: Check out this [notebook ](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb)

#### 2. Structured De-Identification Module:

Introduction of a new annotator to handle de-identification of structured data. it  allows users to define a mapping of columns and their obfuscation policy. Users can also provide dummy data and map them to columns they want to replace values in.

**How to use:**

	obfuscator = StructuredDeidentification \
		(spark,{"NAME":"PATIENT","AGE":"AGE"},
		obfuscateRefSource = "faker")

	obfuscator_df = obfuscator.obfuscateColumns(df)

	obfuscator_df.select("NAME","AGE").show(truncate=False)

**Example:**

Input Data:

Name  | Age
------------- | -------------
Cecilia Chapman|83
Iris Watson    |9  
Bryar Pitts    |98
Theodore Lowe  |16
Calista Wise   |76

Deidentified:

Name  | Age
------------- | -------------
Menne Erdôs|20
 Longin Robinson     |31  
 Flynn Fiedlerová   |50
 John Wakeland  |21
Vanessa Andersson   |12


For more details: Check out this [notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.Clinical_DeIdentification.ipynb).

#### 3. Introducing SOTA relation extraction model using BioBert

A brand-new end-to-end trained BERT model, resulting in massive improvements. Another new annotator (`ReChunkFilter`) is also developed for this new model to allow syntactic features work well with BioBert to extract relations.

**How to use:**

    re_ner_chunk_filter = RENerChunksFilter()\
        .setInputCols(["ner_chunks", "dependencies"])\
        .setOutputCol("re_ner_chunks")\
        .setRelationPairs(pairs)\
        .setMaxSyntacticDistance(4)

    re_model = RelationExtractionDLModel()\
        .pretrained(“redl_temporal_events_biobert”, "en", "clinical/models")\
        .setPredictionThreshold(0.9)\
        .setInputCols(["re_ner_chunks", "sentences"])\
        .setOutputCol("relations")


##### Benchmarks:

**on benchmark datasets**

model                           | Spark NLP ML model | Spark NLP DL model | benchmark
---------------------------------|-----------|-----------|-----------
re_temporal_events_clinical     | 68.29     | 71.0      | 80.2 [1](https://arxiv.org/pdf/2012.08790.pdf)
re_clinical                     | 56.45     | **69.2**      | 68.2      [2](ncbi.nlm.nih.gov/pmc/articles/PMC7153059/)
re_human_pheotype_gene_clinical | -         | **87.9**      | 67.2 [3](https://arxiv.org/pdf/1903.10728.pdf)
re_drug_drug_interaction        | -         | 72.1      | 83.8 [4](https://www.aclweb.org/anthology/2020.knlp-1.4.pdf)
re_chemprot                     | 76.69     | **94.1**      | 83.64 [5](https://www.aclweb.org/anthology/D19-1371.pdf)

**on in-house annotations**

model                           | Spark NLP ML model | Spark NLP DL model
---------------------------------|-----------|-----------
re_bodypart_problem             | 84.58     | 85.7
re_bodypart_procedure           | 61.0      | 63.3
re_date_clinical                | 83.0      | 84.0  
re_bodypart_direction           | 93.5      | 92.5  


For more details: Check out the [notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb) or [modelshub](https://nlp.johnsnowlabs.com/models?tag=relation_extraction).



#### 4. Drug Normalizer:

Standardize units of drugs and handle abbreviations in raw text or drug chunks identified by any NER model. This normalization significantly improves performance of entity resolvers.  

**How to use:**

    drug_normalizer = DrugNormalizer()\
        .setInputCols("document")\
        .setOutputCol("document_normalized")\
        .setPolicy("all") #all/abbreviations/dosages

**Examples:**

`drug_normalizer.transform("adalimumab 54.5 + 43.2 gm”)`

    >>> "adalimumab 97700 mg"

**Changes:** _combine_ `54.5` + `43.2` and _normalize_ `gm` to `mg`

`drug_normalizer.transform("Agnogenic one half cup”)`

    >>> "Agnogenic 0.5 oral solution"

**Changes:** _replace_ `one half` to the `0.5`, _normalize_ `cup` to the `oral solution`

`drug_normalizer.transform("interferon alfa-2b 10 million unit ( 1 ml ) injec”)`

    >>> "interferon alfa - 2b 10000000 unt ( 1 ml ) injection "

**Changes:** _convert_ `10 million unit` to the `10000000 unt`, _replace_ `injec` with `injection`

For more details: Check out this [notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/23.Drug_Normalizer.ipynb)

#### 5. Assertion models to support confidence in output:

Just like NER output, assertion models now also provides _confidence scores_ for each prediction.

chunks  | entities | assertion | confidence
-----------|-----------|------------|-------------
a headache | PROBLEM | present |0.9992
anxious | PROBLEM | conditional |0.9039
alopecia | PROBLEM | absent |0.9992
pain | PROBLEM | absent |0.9238

`.setClasses()` method is deprecated in `AssertionDLApproach`  and users do not need to specify number of classes while training, as it will be inferred from the dataset.


#### 6. New Relation Extraction Models:


We are also releasing new relation extraction models to link the clinical entities to body parts and dates. These models are trained using binary relation extraction approach for better accuracy.

**- re_bodypart_direction :**  Relation Extraction between `Body Part` and `Direction` entities.

**Example:**

**Text:** _“MRI demonstrated infarction in the upper brain stem , left cerebellum and right basil ganglia”_

relations | entity1                     | chunk1     | entity2                     | chunk2        | confidence
-----------|-----------------------------|------------|-----------------------------|---------------|------------
1         | Direction                   | upper      | bodyPart | brain stem    | 0.999
0         | Direction                   | upper      | bodyPart | cerebellum    | 0.999
0         | Direction                   | upper      | bodyPart | basil ganglia | 0.999
0         | bodyPart | brain stem | Direction                   | left          | 0.999
0         | bodyPart | brain stem | Direction                   | right         | 0.999
1         | Direction                   | left       | bodyPart | cerebellum    | 1.0         
0         | Direction                   | left       | bodyPart | basil ganglia | 0.976
0         | bodyPart | cerebellum | Direction                   | right         | 0.953
1         | Direction                   | right      | bodyPart | basil ganglia | 1.0       


**- re_bodypart_problem :** Relation Extraction between `Body Part` and `Problem` entities.

**Example:**

**Text:** _“No neurologic deficits other than some numbness in his left hand.”_

relation | entity1   | chunk1              | entity2                      | chunk2   | confidence  
--|------------------|--------------------|-----------------------------|---------|-----------  
0 | Symptom   | neurologic deficits | bodyPart | hand     |          1
1 | Symptom   | numbness            | bodyPart | hand     |          1


**- re_bodypart_proceduretest :**  Relation Extraction between `Body Part` and `Procedure`, `Test` entities.

**Example:**

**Text:** _“TECHNIQUE IN DETAIL: After informed consent was obtained from the patient and his mother, the chest was scanned with portable ultrasound.”_

relation | entity1                      | chunk1   | entity2   | chunk2              | confidence  
---------|-----------------------------|---------|----------|--------------------|-----------
1 | bodyPart | chest    | Test      | portable ultrasound |    0.999

**-re_date_clinical :** Relation Extraction between `Date` and different clinical entities.   

**Example:**

**Text:** _“This 73 y/o patient had CT on 1/12/95, with progressive memory and cognitive decline since 8/11/94.”_

relations | entity1 | chunk1                                   | entity2 | chunk2  | confidence
-----------|---------|------------------------------------------|---------|---------|------------|
1         | Test    | CT                                       | Date    | 1/12/95 | 1.0
1         | Symptom | progressive memory and cognitive decline | Date    | 8/11/94 | 1.0


**How to use:**

    re_model = RelationExtractionModel()\
        .pretrained("re_bodypart_direction","en","clinical/models")\
        .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
        .setOutputCol("relations")\
        .setMaxSyntacticDistance(4)\
        .setRelationPairs([‘Internal_organ_or_component’, ‘Direction’])



For more details: Check out the [notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.1.Clinical_Relation_Extraction_BodyParts_Models.ipynb) or [modelshub](https://nlp.johnsnowlabs.com/models?tag=relation_extraction).


**New matching scheme for entity resolvers - improved accuracy:** Adding the option to use `cosine similarity` to resolve entities and find closest matches, resulting in better, more semantically correct results.

#### 7. New Resolver Models using `JSL SBERT`:

+ `sbiobertresolve_icd10cm_augmented`

+ `sbiobertresolve_cpt_augmented`

+ `sbiobertresolve_cpt_procedures_augmented`

+ `sbiobertresolve_icd10cm_augmented_billable_hcc`

+ `sbiobertresolve_hcc_augmented`


**Returning auxilary columns mapped to resolutions:**  Chunk entity resolver and sentence entity resolver now returns auxilary data that is mapped the resolutions during training. This will allow users to get multiple resolutions with single model without using any other annotator in the pipeline (In order to get billable codes otherwise there needs to be other modules in the same pipeline)

**Example:**

`sbiobertresolve_icd10cm_augmented_billable_hcc`
**Input Text:** _"bladder cancer"_

idx | chunks | code | resolutions | all_codes | billable | hcc_status | hcc_score | all_distances   
---|---------------|-------|-------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------|--------------------------|--------------------------|---------------------------|---------------------------------------------------
 0 | bladder cancer | C679 | ['bladder cancer', 'suspected bladder cancer', 'cancer in situ of urinary bladder', 'tumor of bladder neck', 'malignant tumour of bladder neck'] | ['C679', 'Z126', 'D090', 'D494', 'C7911'] | ['1', '1', '1', '1', '1'] | ['1', '0', '0', '0', '1'] | ['11', '0', '0', '0', '8'] | ['0.0000', '0.0904', '0.0978', '0.1080', '0.1281'] |

`sbiobertresolve_cpt_augmented`  
**Input Text:** _"ct abdomen without contrast"_

idx|  cpt code |   distance |resolutions                                                       
---:|------:|-----------:|:-------------------------------------------------------------------
0 | 74150 |     0.0802 | Computed tomography, abdomen; without contrast material            |
1 | 65091 |     0.1312 | Evisceration of ocular contents; without implant                   |
2 | 70450 |     0.1323 | Computed tomography, head or brain; without contrast material      |
3 | 74176 |     0.1333 | Computed tomography, abdomen and pelvis; without contrast material |
4 | 74185 |     0.1343 | Magnetic resonance imaging without contrast                        |
5 | 77059 |     0.1343 | Magnetic resonance imaging without contrast                        |

#### 8. New Pretrained Clinical NER Models

+ NER Radiology
**Input Text:** _"Bilateral breast ultrasound was subsequently performed, which demonstrated an ovoid mass measuring approximately 0.5 x 0.5 x 0.4 cm in diameter located within the anteromedial aspect of the left shoulder. This mass demonstrates isoechoic echotexture to the adjacent muscle, with no evidence of internal color flow. This may represent benign fibrous tissue or a lipoma."_

|idx | chunks                | entities                  |
|----|-----------------------|---------------------------|
| 0  | Bilateral             | Direction                 |
| 1  | breast                | BodyPart                  |
| 2  | ultrasound            | ImagingTest               |
| 3  | ovoid mass            | ImagingFindings           |
| 4  | 0.5 x 0.5 x 0.4       | Measurements              |
| 5  | cm                    | Units                     |
| 6  | anteromedial aspect   | Direction                 |
| 7  | left                  | Direction                 |
| 8  | shoulder              | BodyPart                  |
| 9  | mass                  | ImagingFindings           |
| 10 | isoechoic echotexture | ImagingFindings           |
| 11 | muscle                | BodyPart                  |
| 12 | internal color flow   | ImagingFindings           |
| 13 | benign fibrous tissue | ImagingFindings           |
| 14 | lipoma                | Disease_Syndrome_Disorder |


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_7_2">Version 2.7.2</a>
    </li>
    <li>
        <strong>Version 2.7.3</strong>
    </li>
    <li>
        <a href="release_notes_2_7_4">Version 2.7.4</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
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
    <li class="active"><a href="release_notes_2_7_3">2.7.3</a></li>
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