---
layout: docs
header: true
title: Spark NLP for Healthcare Release Notes
permalink: /docs/en/licensed_release_notes
key: docs-licensed-release-notes
modify_date: 2021-02-20
---

# Release Notes Spark NLP Healthcare

### 2.7.4

We are glad to announce that Spark NLP for Healthcare 2.7.4 has been released!  

#### Highlights:

- Introducing a new annotator to extract chunks with NER tags using regex-like patterns: **NerChunker**.
- Introducing two new annotators to filter chunks: **ChunkFilterer** and **AssertionFilterer**.
- Ability to change the entity type in **NerConverterInternal** without using ChunkMerger (`setReplaceDict`).
- In **DeIdentification** model, ability to use `faker` and static look-up lists at the same time randomly in `Obfuscation` mode.
- New **De-Identification NER** model, augmented with synthetic datasets to detect uppercased name entities.
- Bug fixes & general improvements.

#### 1. NerChunker:

Similar to what we used to do in **POSChunker** with POS tags, now we can also extract phrases that fits into a known pattern using the NER tags. **NerChunker** would be quite handy to extract entity groups with neighboring tokens when there is no pretrained NER model to address certain issues. Lets say we want to extract clinical findings and body parts together as a single chunk even if there are some unwanted tokens between. 

**How to use:**

    ner_model = NerDLModel.pretrained("ner_radiology", "en", "clinical/models")\
        .setInputCols("sentence","token","embeddings")\
        .setOutputCol("ner")
                
    ner_chunker = NerChunker().\
        .setInputCols(["sentence","ner"])\
        .setOutputCol("ner_chunk")\
        .setRegexParsers(["<IMAGINGFINDINGS>*<BODYPART>"])

    text = 'She has cystic cyst on her kidney.'

    >> ner tags: [(cystic, B-IMAGINGFINDINGS), (cyst,I-IMAGINGFINDINGS), (kidney, B-BODYPART)
    >> ner_chunk: ['cystic cyst on her kidney']


#### 2. ChunkFilterer:

**ChunkFilterer** will allow you to filter out named entities by some conditions or predefined look-up lists, so that you can feed these entities to other annotators like Assertion Status or Entity Resolvers.  It can be used with two criteria: **isin** and **regex**.

**How to use:**

    ner_model = NerDLModel.pretrained("ner_clinical", "en", "clinical/models")\
          .setInputCols("sentence","token","embeddings")\
          .setOutputCol("ner")

    ner_converter = NerConverter() \
          .setInputCols(["sentence", "token", "ner"]) \
          .setOutputCol("ner_chunk")
          
    chunk_filterer = ChunkFilterer()\
          .setInputCols("sentence","ner_chunk")\
          .setOutputCol("chunk_filtered")\
          .setCriteria("isin") \ 
          .setWhiteList(['severe fever','sore throat'])

    text = 'Patient with severe fever, sore throat, stomach pain, and a headache.'

    >> ner_chunk: ['severe fever','sore throat','stomach pain','headache']
    >> chunk_filtered: ['severe fever','sore throat']


#### 3. AssertionFilterer:

**AssertionFilterer** will allow you to filter out the named entities by the list of acceptable assertion statuses. This annotator would be quite handy if you want to set a white list for the acceptable assertion statuses like `present` or `conditional`; and do not want `absent` conditions get out of your pipeline.

**How to use:**

    clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
      .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
      .setOutputCol("assertion")
    
    assertion_filterer = AssertionFilterer()\
      .setInputCols("sentence","ner_chunk","assertion")\
      .setOutputCol("assertion_filtered")\
      .setWhiteList(["present"])


    text = 'Patient with severe fever and sore throat, but no stomach pain.'

    >> ner_chunk: ['severe fever','sore throat','stomach pain','headache']
    >> assertion_filtered: ['severe fever','sore throat']

 
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



### 2.7.2

We are glad to announce that Spark NLP for Healthcare 2.7.2 has been released !

In this release, we introduce the following features:

+ Far better accuracy for resolving medication terms to RxNorm codes:

  `ondansetron 8 mg tablet' -> '312086`
+ Far better accuracy for resolving diagnosis terms to ICD-10-CM codes:

 `TIA -> transient ischemic attack (disorder)	‘S0690’`
+ New ability to map medications to pharmacological actions (PA):

  `'metformin' -> ‘Hypoglycemic Agents’ `
+ 2 new *greedy* named entity recognition models for medication details:

 `ner_drugs_greedy: ‘magnesium hydroxide 100mg/1ml PO’`

 ` ner_posology _greedy: ‘12 units of insulin lispro’ `

+ New model to *classify the gender* of a patient in a given medical note:

 `'58yo patient with a family history of breast cancer' -> ‘female’ `
+ And starting customized spark sessions with rich parameters

```python
        params = {"spark.driver.memory":"32G",
        "spark.kryoserializer.buffer.max":"2000M",
        "spark.driver.maxResultSize":"2000M"}

        spark = sparknlp_jsl.start(secret, params=params)
```
State-of-the-art accuracy is achieved using new healthcare-tuned BERT Sentence Embeddings (s-Bert). The following sections include more details, metrics, and examples.

#### Named Entity Recognizers for Medications


+ A new medication NER (`ner_drugs_greedy`) that joins the drug entities with neighboring entities such as  `dosage`, `route`, `form` and `strength`; and returns a single entity `drug`.  This greedy NER model would be highly useful if you want to extract a drug with its context and then use it to get a RxNorm code (drugs may get different RxNorm codes based on the dosage and strength information).

###### Metrics

| label  | tp    | fp   | fn   | prec  | rec   | f1    |
|--------|-------|------|------|-------|-------|-------|
| I-DRUG | 37423 | 4179 | 3773 | 0.899 | 0.908 | 0.904 |
| B-DRUG | 29699 | 2090 | 1983 | 0.934 | 0.937 | 0.936 |


+ A new medication NER (`ner_posology_greedy`) that joins the drug entities with neighboring entities such as  `dosage`, `route`, `form` and `strength`.  It also returns all the other medication entities even if not related to (or joined with) a drug.   

Now we have five different medication-related NER models. You can see the outputs from each model below:

Text = ''*The patient was prescribed 1 capsule of Advil 10 mg for 5 days and magnesium hydroxide 100mg/1ml suspension PO. He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night, 12 units of insulin lispro with meals, and metformin 1000 mg two times a day.*''

a. **ner_drugs_greedy**

|   | chunks                           | begin | end | entities |
| - | -------------------------------- | ----- | --- | -------- |
| 0 | 1 capsule of Advil 10 mg         | 27    | 50  | DRUG     |
| 1 | magnesium hydroxide 100mg/1ml PO | 67    | 98  | DRUG     |
| 2 |  40 units of insulin glargine    | 168   | 195 | DRUG     |
| 3 |  12 units of insulin lispro      | 207   | 232 | DRUG     |

b. **ner_posology_greedy**

|    | chunks                           |   begin |   end | entities   |
|---:|:---------------------------------|--------:|------:|:-----------|
|  0 | 1 capsule of Advil 10 mg         |      27 |    50 | DRUG       |
|  1 | magnesium hydroxide 100mg/1ml PO |      67 |    98 | DRUG       |
|  2 | for 5 days                       |      52 |    61 | DURATION   |
|  3 | 40 units of insulin glargine     |     168 |   195 | DRUG       |
|  4 | at night                         |     197 |   204 | FREQUENCY  |
|  5 | 12 units of insulin lispro       |     207 |   232 | DRUG       |
|  6 | with meals                       |     234 |   243 | FREQUENCY  |
|  7 | metformin 1000 mg                |     250 |   266 | DRUG       |
|  8 | two times a day                  |     268 |   282 | FREQUENCY  |

c. **ner_drugs**

|    | chunks              |   begin |   end | entities   |
|---:|:--------------------|--------:|------:|:-----------|
|  0 | Advil               |      40 |    44 | DrugChem   |
|  1 | magnesium hydroxide |      67 |    85 | DrugChem   |
|  2 | metformin           |     261 |   269 | DrugChem   |


d.**ner_posology**

|    | chunks              |   begin |   end | entities   |
|---:|:--------------------|--------:|------:|:-----------|
|  0 | 1                   |      27 |    27 | DOSAGE     |
|  1 | capsule             |      29 |    35 | FORM       |
|  2 | Advil               |      40 |    44 | DRUG       |
|  3 | 10 mg               |      46 |    50 | STRENGTH   |
|  4 | for 5 days          |      52 |    61 | DURATION   |
|  5 | magnesium hydroxide |      67 |    85 | DRUG       |
|  6 | 100mg/1ml           |      87 |    95 | STRENGTH   |
|  7 | PO                  |      97 |    98 | ROUTE      |
|  8 | 40 units            |     168 |   175 | DOSAGE     |
|  9 | insulin glargine    |     180 |   195 | DRUG       |
| 10 | at night            |     197 |   204 | FREQUENCY  |
| 11 | 12 units            |     207 |   214 | DOSAGE     |
| 12 | insulin lispro      |     219 |   232 | DRUG       |
| 13 | with meals          |     234 |   243 | FREQUENCY  |
| 14 | metformin           |     250 |   258 | DRUG       |
| 15 | 1000 mg             |     260 |   266 | STRENGTH   |
| 16 | two times a day     |     268 |   282 | FREQUENCY  |

e. **ner_drugs_large**

|    | chunks                            |   begin |   end | entities   |
|---:|:----------------------------------|--------:|------:|:-----------|
|  0 | Advil 10 mg                       |      40 |    50 | DRUG       |
|  1 | magnesium hydroxide 100mg/1ml PO. |      67 |    99 | DRUG       |
|  2 | insulin glargine                  |     180 |   195 | DRUG       |
|  3 | insulin lispro                    |     219 |   232 | DRUG       |
|  4 | metformin 1000 mg                 |     250 |   266 | DRUG       |


#### Patient Gender Classification

This model detects the gender of the patient in the clinical document. It can classify the documents into `Female`, `Male` and `Unknown`.

We release two models:

+ 'Classifierdl_gender_sbert' (more accurate, works with licensed `sbiobert_base_cased_mli`)

+ 'Classifierdl_gender_biobert' (works with `biobert_pubmed_base_cased`)

The models are trained on more than four thousands clinical documents (radiology reports, pathology reports, clinical visits etc.), annotated internally.

###### Metrics `(Classifierdl_gender_sbert)`

|        | precision | recall | f1-score | support |
| ------ | --------- | ------ | -------- | ------- |
| Female | 0.9224    | 0.8954 | 0.9087   | 239     |
| Male   | 0.7895    | 0.8468 | 0.8171   | 124     |

Text= ''*social history: shows that  does not smoke cigarettes or drink alcohol, lives in a nursing home.
family history: shows a family history of breast cancer.*''

```python
gender_classifier.annotate(text)['class'][0]
>> `Female`
```

See this [Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/21_Gender_Classifier.ipynb) notebook for further details.

a. **classifierdl_gender_sbert**

```python

document = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

sbert_embedder = BertSentenceEmbeddings\
    .pretrained("sbiobert_base_cased_mli", 'en', 'clinical/models')\
    .setInputCols(["document"])\
    .setOutputCol("sentence_embeddings")\
    .setMaxSentenceLength(512)

gender_classifier = ClassifierDLModel\
    .pretrained('classifierdl_gender_sbert', 'en', 'clinical/models') \
    .setInputCols(["document", "sentence_embeddings"]) \
    .setOutputCol("class")

gender_pred_pipeline = Pipeline(
    stages = [
       document,
       sbert_embedder,
       gender_classifier
            ])
```
b. **classifierdl_gender_biobert**

```python
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

clf_tokenizer = Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")\

biobert_embeddings = BertEmbeddings().pretrained('biobert_pubmed_base_cased') \
    .setInputCols(["document",'token'])\
    .setOutputCol("bert_embeddings")

biobert_embeddings_avg = SentenceEmbeddings() \
    .setInputCols(["document", "bert_embeddings"]) \
    .setOutputCol("sentence_bert_embeddings") \
    .setPoolingStrategy("AVERAGE")

genderClassifier = ClassifierDLModel.pretrained('classifierdl_gender_biobert', 'en', 'clinical/models') \
    .setInputCols(["document", "sentence_bert_embeddings"]) \
    .setOutputCol("gender")

gender_pred_pipeline = Pipeline(
   stages = [
       documentAssembler,
       clf_tokenizer,
       biobert_embeddings,
       biobert_embeddings_avg,
       genderClassifier
   ])

   ```
#### New ICD10CM and RxCUI resolvers powered by s-Bert embeddings

The advent of s-Bert sentence embeddings changed the landscape of Clinical Entity Resolvers completely in Spark NLP. Since s-Bert is already tuned on [MedNLI](https://physionet.org/content/mednli/) (medical natural language inference) dataset, it is now capable of populating the chunk embeddings in a more precise way than before.

We now release two new resolvers:

+ `sbiobertresolve_icd10cm_augmented` (augmented with synonyms, four times richer than previous resolver accuracy:

    `73% for top-1 (exact match), 89% for top-5 (previous accuracy was 59% and 64% respectively)`

+ `sbiobertresolve_rxcui` (extract RxNorm concept unique identifiers to map with ATC or durg families)
accuracy:

    `71% for top-1 (exact match), 72% for top-5
(previous accuracy was 22% and 41% respectively)`

a. **ICD10CM augmented resolver**

Text = "*This is an 82 year old male with a history of prior tobacco use , hypertension , chronic renal insufficiency , COPD , gastritis , and TIA who initially presented to Braintree with a non-ST elevation MI and Guaiac positive stools , transferred to St . Margaret\'s Center for Women & Infants for cardiac catheterization with PTCA to mid LAD lesion complicated by hypotension and bradycardia requiring Atropine , IV fluids and transient dopamine possibly secondary to vagal reaction , subsequently transferred to CCU for close monitoring , hemodynamically stable at the time of admission to the CCU .* "

|    | chunk                       |   begin |   end | code   | term                                                     |
|---:|:----------------------------|--------:|------:|:-------|:---------------------------------------------------------|
|  0 | hypertension                |      66 |    77 | I10    | hypertension                                             |
|  1 | chronic renal insufficiency |      81 |   107 | N189   | chronic renal insufficiency                              |
|  2 | COPD                        |     111 |   114 | J449   | copd - chronic obstructive pulmonary disease             |
|  3 | gastritis                   |     118 |   126 | K2970  | gastritis                                                |
|  4 | TIA                         |     134 |   136 | S0690  | transient ischemic attack (disorder)		 |
|  5 | a non-ST elevation MI       |     180 |   200 | I219   | silent myocardial infarction (disorder)                  |
|  6 | Guaiac positive stools      |     206 |   227 | K921   | guaiac-positive stools                                   |
|  7 | mid LAD lesion              |     330 |   343 | I2102  | stemi involving left anterior descending coronary artery |
|  8 | hypotension                 |     360 |   370 | I959   | hypotension                                              |
|  9 | bradycardia                 |     376 |   386 | O9941  | bradycardia                                              |


b. **RxCUI resolver**

Text= "*He was seen by the endocrinology service and she was discharged on 50 mg of eltrombopag oral at night, 5 mg amlodipine with meals, and metformin 1000 mg two times a day .* "

|    | chunk                     |   begin |   end |   code | term                                        |
|---:|:--------------------------|--------:|------:|-------:|:--------------------------------------------|
|  0 | 50 mg of eltrombopag oral |      67 |    91 | 825427 | eltrombopag 50 MG Oral Tablet               |
|  1 | 5 mg amlodipine           |     103 |   117 | 197361 | amlodipine 5 MG Oral Tablet                 |
|  2 | metformin 1000 mg         |     135 |   151 | 861004 | metformin hydrochloride 1000 MG Oral Tablet |

Using this new resolver and some other resources like Snomed Resolver, RxTerm, MESHPA and ATC dictionary, you can link the drugs to the pharmacological actions (PA), ingredients and the disease treated with that.

###### Code sample:

(after getting the chunk from ChunkConverter)


 ```python

c2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")

sbert_embedder = BertSentenceEmbeddings\
    .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
    .setInputCols(["ner_chunk_doc"])\
    .setOutputCol("sbert_embeddings")

icd10_resolver = SentenceEntityResolverModel.pretrained("sbiobertresolve_icd10cm_augmented","en", "clinical/models") \
    .setInputCols(["ner_chunk", "sbert_embeddings"]) \
    .setOutputCol("icd10cm_code")\
    .setDistanceFunction("EUCLIDEAN")
```
See the [notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb#scrollTo=VtDWAlnDList) for details.


### 2.7.1

We are glad to announce that Spark NLP for Healthcare 2.7.1 has been released !

In this release, we introduce the following features:

#### 1. Sentence BioBert and Bluebert Transformers that are fine tuned on [MedNLI](https://physionet.org/content/mednli/) dataset.

Sentence Transformers offers a framework that provides an easy method to compute dense vector representations for sentences and paragraphs (also known as sentence embeddings). The models are based on BioBert and BlueBert, and are tuned specifically to meaningful sentence embeddings such that sentences with similar meanings are close in vector space. These are the first PyTorch based models we managed to port into Spark NLP.

Here is how you can load these:
```python
sbiobert_embeddins = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")
```
```python
sbluebert_embeddins = BertSentenceEmbeddings\
     .pretrained("sbluebert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")
```

#### 2. SentenceEntityResolvers powered by s-Bert embeddings.

The advent of s-Bert sentence embeddings changed the landscape of Clinical Entity Resolvers completely in Spark NLP. Since s-Bert is already tuned on MedNLI (medical natural language inference) dataset, it is now capable of populating the chunk embeddings in a more precise way than before.

Using sbiobert_base_cased_mli, we trained the following Clinical Entity Resolvers:

sbiobertresolve_icd10cm  
sbiobertresolve_icd10pcs  
sbiobertresolve_snomed_findings (with clinical_findings concepts from CT version)  
sbiobertresolve_snomed_findings_int  (with clinical_findings concepts from INT version)  
sbiobertresolve_snomed_auxConcepts (with Morph Abnormality, Procedure, Substance, Physical Object, Body Structure concepts from CT version)  
sbiobertresolve_snomed_auxConcepts_int  (with Morph Abnormality, Procedure, Substance, Physical Object, Body Structure concepts from INT version)  
sbiobertresolve_rxnorm  
sbiobertresolve_icdo  
sbiobertresolve_cpt  

Code sample:

(after getting the chunk from ChunkConverter)

```python
c2doc = Chunk2Doc().setInputCols("ner_chunk").setOutputCol("ner_chunk_doc")
sbert_embedder = BertSentenceEmbeddings\
     .pretrained("sbiobert_base_cased_mli",'en','clinical/models')\
     .setInputCols(["ner_chunk_doc"])\
     .setOutputCol("sbert_embeddings")

snomed_ct_resolver = SentenceEntityResolverModel
 .pretrained("sbiobertresolve_snomed_findings","en", "clinical/models") \
 .setInputCols(["ner_chunk", "sbert_embeddings"]) \
 .setOutputCol("snomed_ct_code")\
 .setDistanceFunction("EUCLIDEAN")
 ```

Output:

|    | chunks                      |   begin |   end |      code | resolutions
|  2 | COPD  				 |     113 |   116 |  13645005 | copd - chronic obstructive pulmonary disease
|  8 | PTCA                        |     324 |   327 | 373108000 | post percutaneous transluminal coronary angioplasty (finding)
| 16 | close monitoring            |     519 |   534 | 417014005 | on examination - vigilance


See the [notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb#scrollTo=VtDWAlnDList) for details.

#### 3. We are releasing the following pretrained clinical NER models:

ner_drugs_large   
(trained with medications dataset, and extracts drugs with the dosage, strength, form and route at once as a single entity; entities: drug)  
ner_deid_sd_large  
(extracts PHI entities, trained with augmented dataset)  
ner_anatomy_coarse  
(trained with enriched anatomy NER dataset; entities: anatomy)  
ner_anatomy_coarse_biobert  
chunkresolve_ICD10GM_2021 (German ICD10GM resolver)


We are also releasing two new NER models:

ner_aspect_based_sentiment  
(extracts positive, negative and neutral aspects about restaurants from the written feedback given by reviewers. )  
ner_financial_contract  
(extract financial entities from contracts. See the [notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/19.Financial_Contract_NER.ipynb) for details.)  


### 2.7.0

We are glad to announce that Spark NLP for Healthcare 2.7 has been released !

In this release, we introduce the following features:

#### 1. Text2SQL

Text2SQL Annotator that translates natural language text into SQL queries against a predefined database schema, which is one of the
most sought-after features of NLU. With the help of a pretrained text2SQL model, you will be able to query your database without writing a SQL query:

Example 1

Query:
What is the name of the nurse who has the most appointments?

Generated
SQL query from the model:

```sql
SELECT T1.Name  
FROM Nurse AS T1  
JOIN Appointment AS T2 ON T1.EmployeeID = T2.PrepNurse  
GROUP BY T2.prepnurse  
ORDER BY count(*) DESC  
LIMIT 1  
```

Response:  

|    | Name           |
|---:|:---------------|
|  0 | Carla Espinosa |

Example 2

Query:
How many patients do each physician take care of? List their names and number of patients they take care of.

Generated
SQL query from the model:

```sql
SELECT T1.Name,  
count(*)  
FROM Physician AS T1  
JOIN Patient AS T2 ON T1.EmployeeID = T2.PCP  
GROUP BY T1.Name  
```

Response:   

|    | Name             |   count(*) |
|---:|:-----------------|-----------:|
|  0 | Christopher Turk |          1 |
|  1 | Elliot Reid      |          2 |
|  2 | John Dorian      |          1 |


For now, it only comes with one pretrained model (trained on Spider
dataset) and new pretrained models will be released soon.

Check out the
Colab notebook to  see more examples and run on your data.


#### 2. SentenceEntityResolvers

In addition to ChunkEntityResolvers, we now release our first BioBert-based entity resolvers using the SentenceEntityResolver
annotator. It’s
fully trainable and comes with several pretrained entity resolvers for the following medical terminologies:

CPT: `biobertresolve_cpt`  
ICDO: `biobertresolve_icdo`  
ICD10CM: `biobertresolve_icd10cm`  
ICD10PCS: `biobertresolve_icd10pcs`  
LOINC: `biobertresolve_loinc`  
SNOMED_CT (findings): `biobertresolve_snomed_findings`  
SNOMED_INT (clinical_findings): `biobertresolve_snomed_findings_int`    
RXNORM (branded and clinical drugs): `biobertresolve_rxnorm_bdcd`  

Example:
```python
text = 'He has a starvation ketosis but nothing significant for dry oral mucosa'
df = get_icd10_codes (light_pipeline_icd10, 'icd10cm_code', text)
```

|    | chunks               |   begin |   end | code |
|---:|:---------------------|--------:|------:|:-------|
|  0 | a starvation ketosis |       7 |    26 | E71121 |                                                                                                                                                   
|  1 | dry oral mucosa      |      66 |    80 | K136   |


Check out the Colab notebook to  see more examples and run on your data.

You can also train your own entity resolver using any medical terminology like MedRa and UMLS. Check this notebook to
learn more about training from scratch.


#### 3. ChunkMerge Annotator

In order to use multiple NER models in the same pipeline, Spark NLP Healthcare has ChunkMerge Annotator that is used to return entities from each NER
model by overlapping. Now it has a new parameter to avoid merging overlapping entities (setMergeOverlapping)
to return all the entities regardless of char indices. It will be quite useful to analyze what every NER module returns on the same text.


#### 4. Starting SparkSession


We now support starting SparkSession with a different version of the open source jar and not only the one it was built
against by `sparknlp_jsl.start(secret, public="x.x.x")` for extreme cases.


#### 5. Biomedical NERs

We are releasing 3 new biomedical NER models trained with clinical embeddings (all one single entity models)  

`ner_bacterial_species` (comprising of Linneaus and Species800 datasets)  
`ner_chemicals` (general purpose and bio chemicals, comprising of BC4Chem and BN5CDR-Chem)  
`ner_diseases_large` (comprising of ner_disease, NCBI_Disease and BN5CDR-Disease)  

We are also releasing the biobert versions of the several clinical NER models stated below:  
`ner_clinical_biobert`  
`ner_anatomy_biobert`  
`ner_bionlp_biobert`  
`ner_cellular_biobert`  
`ner_deid_biobert`  
`ner_diseases_biobert`  
`ner_events_biobert`  
`ner_jsl_biobert`  
`ner_chemprot_biobert`  
`ner_human_phenotype_gene_biobert`  
`ner_human_phenotype_go_biobert`  
`ner_posology_biobert`  
`ner_risk_factors_biobert`  


Metrics (micro averages excluding O’s):

|    | model_name                        |   clinical_glove_micro |   biobert_micro |
|---:|:----------------------------------|-----------------------:|----------------:|
|  0 | ner_chemprot_clinical             |               0.816 |      0.803 |
|  1 | ner_bionlp                        |               0.748 |      0.808  |
|  2 | ner_deid_enriched                 |               0.934 |      0.918  |
|  3 | ner_posology                      |               0.915 |      0.911    |
|  4 | ner_events_clinical               |               0.801 |      0.809  |
|  5 | ner_clinical                      |               0.873 |      0.884  |
|  6 | ner_posology_small                |               0.941 |            |
|  7 | ner_human_phenotype_go_clinical   |               0.922 |      0.932  |
|  8 | ner_drugs                         |               0.964 |           |
|  9 | ner_human_phenotype_gene_clinical |               0.876 |      0.870  |
| 10 | ner_risk_factors                  |               0.728 |            |
| 11 | ner_cellular                      |               0.813 |      0.812  |
| 12 | ner_posology_large                |               0.921 |            |
| 13 | ner_anatomy                       |               0.851 |      0.831  |
| 14 | ner_deid_large                    |               0.942 |            |
| 15 | ner_diseases                      |               0.960 |      0.966  |


In addition to these, we release two new German NER models:

`ner_healthcare_slim` ('TIME_INFORMATION', 'MEDICAL_CONDITION',  'BODY_PART',  'TREATMENT', 'PERSON', 'BODY_PART')  
`ner_traffic` (extract entities regarding traffic accidents e.g. date, trigger, location etc.)  

#### 6. PICO Classifier

Successful evidence-based medicine (EBM) applications rely on answering clinical questions by analyzing large medical literature databases. In order to formulate
a well-defined, focused clinical question, a framework called PICO is widely used, which identifies the sentences in a given medical text that belong to the four components: Participants/Problem (P)  (e.g., diabetic patients), Intervention (I) (e.g., insulin),
Comparison (C) (e.g., placebo)  and Outcome (O) (e.g., blood glucose levels).

Spark NLP now introduces a pretrained PICO Classifier that
is trained with Biobert embeddings.

Example:

```python
text = “There appears to be no difference in smoking cessation effectiveness between 1mg and 0.5mg varenicline.”
pico_lp_pipeline.annotate(text)['class'][0]

ans: CONCLUSIONS
```

### 2.6.2

#### Overview
We are very happy to announce that version 2.6.2 of Spark NLP Enterprise is ready to be installed and used.
We are making available Named Entity Recognition, Sentence Classification and Entity Resolution models to analyze Adverse Drug Events in natural language text from clinical domains.

#### Models

##### NERs
We are pleased to announce that we have a brand new named entity recognition (NER) model for Adverse Drug Events (ADE) to extract ADE and DRUG entities from a given text.

ADE NER will have four versions in the library, trained with different size of word embeddings:

`ner_ade_bioert` (768d Bert embeddings)  
`ner_ade_clinicalbert` (768d Bert embeddings)  
`ner_ade_clinical` (200d clinical embeddings)  
`ner_ade_healthcare` (100d healthcare embeddings)  

More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb)

We are also releasing our first clinical pretrained classifier for ADE classification tasks. This new ADE classifier is trained on various ADE datasets, including the mentions in tweets to represent the daily life conversations as well. So it works well on the texts coming from academic context, social media and clinical notes. It’s trained with `Clinical Biobert` embeddings, which is the most powerful contextual language model in the clinical domain out there.

##### Classifiers
ADE classifier will have two versions in the library, trained with different Bert embeddings:

`classifierdl_ade_bioert` (768d BioBert embeddings)  
`classifierdl_adee_clinicalbert` (768d ClinicalBert embeddings)  

More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/16.Adverse_Drug_Event_ADE_NER_and_Classifier.ipynb)

##### Pipeline
By combining ADE NER and Classifier, we are releasing a new pretrained clinical pipeline for ADE tasks to save you from building pipelines from scratch. Pretrained pipelines are already fitted using certain annotators and transformers according to various use cases and you can use them as easy as follows:
```python
pipeline = PretrainedPipeline('explain_clinical_doc_ade', 'en', 'clinical/models')

pipeline.annotate('my string')
```
`explain_clinical_doc_ade` is bundled with `ner_ade_clinicalBert`, and `classifierdl_ade_clinicalBert`. It can extract ADE and DRUG clinical entities, and then assign ADE status to a text (`True` means ADE, `False` means not related to ADE).

More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb)


##### Entity Resolver
We are releasing the first Entity Resolver for `Athena` (Automated Terminology Harmonization, Extraction and Normalization for Analytics, http://athena.ohdsi.org/) to extract concept ids via standardized medical vocabularies. For now, it only supports `conditions` section and can be used to map the clinical conditions with the corresponding standard terminology and then get the concept ids to store them in various database schemas.
It is named as `chunkresolve_athena_conditions_healthcare`.

We added slim versions of several clinical NER models that are trained with 100d healthcare word embeddings, which is lighter and smaller in size.

`ner_healthcare`
`assertion_dl_healthcare`
`ner_posology_healthcare`
`ner_events_healthcare`

##### Graph Builder
Spark NLP Licensed version has several DL based annotators (modules) such as NerDL, AssertionDL, RelationExtraction and GenericClassifier, and they are all based on Tensorflow (tf) with custom graphs. In order to make the creating and customizing the tf graphs for these models easier for our licensed users, we added a graph builder to the Python side of the library. Now you can customize your graphs and use them in the respected models while training a new DL model.

```python
from sparknlp_jsl.training import tf_graph

tf_graph.build("relation_extraction",build_params={"input_dim": 6000, "output_dim": 3, 'batch_norm':1, "hidden_layers": [300, 200], "hidden_act": "relu", 'hidden_act_l2':1}, model_location=".", model_filename="re_with_BN")
```
More information and examples [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/17.Graph_builder_for_DL_models.ipynb)


### 2.6.0

#### Overview

We are honored to announce that Spark NLP Enterprise 2.6.0 has been released.
The first time ever, we release three pretrained clinical pipelines to save you from building pipelines from scratch. Pretrained pipelines are already fitted using certain annotators and transformers according to various use cases.
The first time ever, we are releasing 3 licensed German models for healthcare and Legal domains.

</div><div class="h3-box" markdown="1">

#### Models

##### Pretrained Pipelines:

The first time ever, we release three pretrained clinical pipelines to save you from building pipelines from scratch.
Pretrained pipelines are already fitted using certain annotators and transformers according to various use cases and you can use them as easy as follows:

```python
pipeline = PretrainedPipeline('explain_clinical_doc_carp', 'en', 'clinical/models')

pipeline.annotate('my string')
```

Pipeline descriptions:

- ```explain_clinical_doc_carp``` a pipeline with ner_clinical, assertion_dl, re_clinical and ner_posology. It will extract clinical and medication entities, assign assertion status and find relationships between clinical entities.

- ```explain_clinical_doc_era``` a pipeline with ner_clinical_events, assertion_dl and re_temporal_events_clinical. It will extract clinical entities, assign assertion status and find temporal relationships between clinical entities.

- ```recognize_entities_posology``` a pipeline with ner_posology. It will only extract medication entities.

More information and examples are available here: https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb.

</div><div class="h3-box" markdown="1">

#### Pretrained Named Entity Recognition and Relationship Extraction Models (English)

RE models:

```
re_temporal_events_clinical
re_temporal_events_enriched_clinical
re_human_phenotype_gene_clinical
re_drug_drug_interaction_clinical
re_chemprot_clinical
```
NER models:

```
ner_human_phenotype_gene_clinical
ner_human_phenotype_go_clinical
ner_chemprot_clinical
```
More information and examples here:
https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb

</div><div class="h3-box" markdown="1">

#### Pretrained Named Entity Recognition and Relationship Extraction Models (German)

The first time ever, we are releasing 3 licensed German models for healthcare and Legal domains.

- ```German Clinical NER``` model for 19 clinical entities

- ```German Legal NER``` model for 19 legal entities

- ```German ICD-10GM```

More information and examples here:

https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/14.German_Healthcare_Models.ipynb

https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/15.German_Legal_Model.ipynb

</div><div class="h3-box" markdown="1">

#### Other Pretrained Models

We now have Named Entity Disambiguation model out of the box.

Disambiguation models map words of interest, such as names of persons, locations and companies, from an input text document to corresponding unique entities in a target Knowledge Base (KB).

https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/12.Named_Entity_Disambiguation.ipynb

Due to ongoing requests about Clinical Entity Resolvers, we release a notebook to let you see how to train an entity resolver using an open source dataset based on Snomed.

https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/13.Snomed_Entity_Resolver_Model_Training.ipynb

</div><div class="h3-box" markdown="1">

### 2.5.5

#### Overview

We are very happy to release Spark NLP for Healthcare 2.5.5 with a new state-of-the-art RelationExtraction annotator to identify relationships between entities coming from our pretrained NER models.
This is also the first release to support Relation Extraction with the following two (2) models: `re_clinical` and `re_posology` in the `clinical/models` repository.
We also include multiple bug fixes as usual.

</div><div class="h3-box" markdown="1">

#### New Features

* RelationExtraction annotator that receives `WORD_EMBEDDINGS`, `POS`, `CHUNK`, `DEPENDENCY` and returns the CATEGORY of the relationship and a confidence score.

</div><div class="h3-box" markdown="1">

#### Enhancements

* AssertionDL Annotator now keeps logs of the metrics while training
* DeIdentification now has a default behavior of merging entities close in Levenshtein distance with `setConsistentObfuscation` and `setSameEntityThreshold` params.
* DeIdentification now has a specific parameter `setObfuscateDate` to obfuscate dates (which will be otherwise just masked). The only formats obfuscated when the param is true will be the ones present in `dateFormats` param.
* NerConverterInternal now has a `greedyMode` param that will merge all contiguous tags of the same type regardless of boundary tags like "B","E","S".
* AnnotationToolJsonReader includes `mergeOverlapping` parameter to merge (or not) overlapping entities from the Annotator jsons i.e. not included in the assertion list.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* DeIdentification documentation bug fix (typo)
* DeIdentification training bug fix in obfuscation dictionary
* IOBTagger now has the correct output type `NAMED_ENTITY`

</div><div class="h3-box" markdown="1">

#### Deprecations

* EnsembleEntityResolver has been deprecated

Models

* We have 2 new `english` Relationship Extraction model for Clinical and Posology NERs:
   - `re_clinical`: with `ner_clinical` and `embeddings_clinical`
   - `re_posology`: with `ner_posology` and `embeddings_clinical`

</div><div class="h3-box" markdown="1">

### 2.5.3

#### Overview

We are pleased to announce the release of Spark NLP for Healthcare 2.5.3.
This time we include four (4) new Annotators: FeatureAssembler, GenericClassifier, Yake Keyword Extractor and NerConverterInternal.
We also include helper classes to read datasets from CodiEsp and Cantemist Spanish NER Challenges.
This is also the first release to support the following models: `ner_diag_proc` (spanish), `ner_neoplasms` (spanish), `ner_deid_enriched` (english).
We have also included Bugifxes and Enhancements for AnnotationToolJsonReader and ChunkMergeModel.

</div><div class="h3-box" markdown="1">

#### New Features

* FeatureAssembler Transformer: Receives a list of column names containing numerical arrays and concatenates them to form one single `feature_vector` annotation
* GenericClassifier Annotator: Receives a `feature_vector` annotation and outputs a `category` annotation
* Yake Keyword Extraction Annotator: Receives a `token` annotation and outputs multi-token `keyword` annotations
* NerConverterInternal Annotator: Similar to it's open source counterpart in functionality, performs smarter extraction for complex tokenizations and confidence calculation
* Readers for CodiEsp and Cantemist Challenges

</div><div class="h3-box" markdown="1">

#### Enhancements

* AnnotationToolJsonReader includes parameter for preprocessing pipeline (from Document Assembling to Tokenization)
* AnnotationToolJsonReader includes parameter to discard specific entity types

</div><div class="h3-box" markdown="1">

#### Bugfixes

* ChunkMergeModel now prioritizes highest number of different entities when coverage is the same

</div><div class="h3-box" markdown="1">

#### Models

* We have 2 new `spanish` models for Clinical Entity Recognition: `ner_diag_proc` and `ner_neoplasms`
* We have a new `english` Named Entity Recognition model for deidentification: `ner_deid_enriched`

</div><div class="h3-box" markdown="1">

### 2.5.2

#### Overview

We are really happy to bring you Spark NLP for Healthcare 2.5.2, with a couple new features and several enhancements in our existing annotators.
This release was mainly dedicated to generate adoption in our AnnotationToolJsonReader, a connector that provide out-of-the-box support for out Annotation Tool and our practices.
Also the ChunkMerge annotator has ben provided with extra functionality to remove entire entity types and to modify some chunk's entity type
We also dedicated some time in finalizing some refactorization in DeIdentification annotator, mainly improving type consistency and case insensitive entity dictionary for obfuscation.
Thanks to the community for all the feedback and suggestions, it's really comfortable to navigate together towards common functional goals that keep us agile in the SotA.

</div><div class="h3-box" markdown="1">

#### New Features

* Brand new IOBTagger Annotator
* NerDL Metrics provides an intuitive DataFrame API to calculate NER metrics at tag (token) and entity (chunk) level

</div><div class="h3-box" markdown="1">

#### Enhancements

* AnnotationToolJsonReader includes parameters for document cleanup, sentence boundaries and tokenizer split chars
* AnnotationToolJsonReader uses the task title if present and uses IOBTagger annotator
* AnnotationToolJsonReader has improved alignment in assertion train set generation by using an `alignTol` parameter as tollerance in chunk char alignment
* DeIdentification refactorization: Improved typing and replacement logic, case insensitive entities for obfuscation
* ChunkMerge Annotator now handles:
 - Drop all chunks for an entity
 - Replace entity name
 - Change entity type for a specific (chunk, entity) pair
 - Drop specific (chunk, entity) pairs
* `caseSensitive` param to EnsembleEntityResolver
* Output logs for AssertionDLApproach loss
* Disambiguator is back with improved dependency management

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Bugfix in python when Annotators shared domain parts across public and internal
* Bugfix in python when ChunkMerge annotator was loaded from disk
* ChunkMerge now weights the token coverage correctly when multiple multi-token entities overlap

</div><div class="h3-box" markdown="1">

### 2.5.0

#### Overview

We are happy to bring you Spark NLP for Healthcare 2.5.0 with new Annotators, Models and Data Readers.
Model composition and iteration is now faster with readers and annotators designed for real world tasks.
We introduce ChunkMerge annotator to combine all CHUNKS extracted by different Entity Extraction Annotators.
We also introduce an Annotation Reader for JSL AI Platform's Annotation Tool.
This release is also the first one to support the models: `ner_large_clinical`, `ner_events_clinical`, `assertion_dl_large`, `chunkresolve_loinc_clinical`, `deidentify_large`
And of course we have fixed some bugs.

</div><div class="h3-box" markdown="1">

#### New Features

* AnnotationToolJsonReader is a new class that imports a JSON from AI Platform's Annotation Tool an generates NER and Assertion training datasets
* ChunkMerge Annotator is a new functionality that merges two columns of CHUNKs handling overlaps with a very straightforward logic: max coverage, max # entities
* ChunkMerge Annotator handles inputs from NerDLModel, RegexMatcher, ContextualParser, TextMatcher
* A DeIdentification pretrained model can now work in 'mask' or 'obfuscate' mode

</div><div class="h3-box" markdown="1">

#### Enhancements

* DeIdentification Annotator has a more consistent API:
    * `mode` param with values ('mask'l'obfuscate') to drive its behavior
    * `dateFormats` param a list of string values to to select which `dateFormats` to obfuscate (and which to just mask)
* DeIdentification Annotator no longer automatically obfuscates dates. Obfuscation is now driven by `mode` and `dateFormats` params
* A DeIdentification pretrained model can now work in 'mask' or 'obfuscate' mode

</div><div class="h3-box" markdown="1">

#### Bugfixes

* DeIdentification Annotator now correctly deduplicates protected entities coming from NER / Regex
* DeIdentification Annotator now indexes chunks correctly after merging them
* AssertionDLApproach Annotator can now be trained with the graph in any folder specified by setting `graphFolder` param
* AssertionDLApproach now has the `setClasses` param setter in Python wrapper
* JVM Memory and Kryo Max Buffer size increased to 32G and 2000M respectively in `sparknlp_jsl.start(secret)` function

</div><div class="h3-box" markdown="1">

### 2.4.6

#### Overview

We release Spark NLP for Healthcare 2.4.6 to fix some minor bugs.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Updated IDF value calculation to be probabilistic based log[(N - df_t) / df_t + 1] as opposed to log[N / df_t]
* TFIDF cosine distance was being calculated with the rooted norms rather than with the original squared norms
* Validation of label cols is now performed at the beginning of EnsembleEntityResolver
* Environment Variable for License value named jsl.settings.license
* Now DocumentLogRegClassifier can be serialized from Python (bug introduced with the implementation of RecursivePipelines, LazyAnnotator attribute)

</div><div class="h3-box" markdown="1">

### 2.4.5

#### Overview

We are glad to announce Spark NLP for Healthcare 2.4.5. As a new feature we are happy to introduce our new EnsembleEntityResolver which allows our Entity Resolution architecture to scale up in multiple orders of magnitude and handle datasets of millions of records on a sub-log computation increase
We also enhanced our ChunkEntityResolverModel with 5 new distance calculations with weighting-array and aggregation-strategy params that results in more levers to finetune its performance against a given dataset.

</div><div class="h3-box" markdown="1">

#### New Features

* EnsembleEntityResolver consisting of an integrated TFIDF-Logreg classifier in the first layer + Multiple ChunkEntityResolvers in the second layer (one per each class)
* Five (5) new distances calculations for ChunkEntityResolver, namely:
    - Token Based: TFIDF-Cosine, Jaccard, SorensenDice
    - Character Based: JaroWinkler and Levenshtein
* Weight parameter that works as a multiplier for each distance result to be considered during their aggregation
* Three (3) aggregation strategies for the enabled distance in a particular instance, namely: AVERAGE, MAX and MIN

</div><div class="h3-box" markdown="1">

#### Enhancements

* ChunkEntityResolver can now compute distances over all the `neighbours` found and return the metadata just for the best `alternatives` that meet the `threshold`;
before it would calculate them over the neighbours and return them all in the metadata
* ChunkEntityResolver now has an `extramassPenalty` parameter to accoun for penalization of token-length difference in compared strings
* Metadata for the ChunkEntityResolver has been updated accordingly to reflect all new features
* StringDistances class has been included in utils to aid in the calculation and organization of different types of distances for Strings
* HasFeaturesJsl trait has been included to support the serialization of Features including [T] <: AnnotatorModel[T] types

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Frequency calculation for WMD in ChunkEntityResolver has been adjusted to account for real word count representation
* AnnotatorType for DocumentLogRegClassifier has been changed to CATEGORY to align with classifiers in Open Source library

</div><div class="h3-box" markdown="1">

#### Deprecations

* Legacy EntityResolver{Approach, Model} classes have been deprecated in favor of ChunkEntityResolver classes
* ChunkEntityResolverSelector classes has been deprecated in favor of EnsembleEntityResolver

</div><div class="h3-box" markdown="1">

### 2.4.2

#### Overview

We are glad to announce Spark NLP for Healthcare 2.4.2. As a new feature we are happy to introduce our new Disambiguation Annotator,
which will let the users resolve different kind of entities based on Knowledge bases provided in the form of Records in a RocksDB database.
We also enhanced / fixed DocumentLogRegClassifier, ChunkEntityResolverModel and ChunkEntityResolverSelector Annotators.

</div><div class="h3-box" markdown="1">

#### New Features

* Disambiguation Annotator (NerDisambiguator and NerDisambiguatorModel) which accepts annotator types CHUNK and SENTENCE_EMBEDDINGS and
returns DISAMBIGUATION annotator type. This output annotation type includes all the matches in the result and their similarity scores in the metadata.

</div><div class="h3-box" markdown="1">

#### Enhancements

* ChunkEntityResolver Annotator now supports both EUCLIDEAN and COSINE distance for the KNN search and WMD calculation.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Fixed a bug in DocumentLogRegClassifier Annotator to support its serialization to disk.
* Fixed a bug in ChunkEntityResolverSelector Annotator to group by both SENTENCE and CHUNK at the time of forwarding tokens and embeddings to the lazy annotators.
* Fixed a bug in ChunkEntityResolverModel in which the same exact embeddings was not included in the neighbours.

</div><div class="h3-box" markdown="1">

### 2.4.1

#### Overview

Introducing Spark NLP for Healthcare 2.4.1 after all the feedback we received in the form of issues and suggestions on our different communication channels.
Even though 2.4.0 was very stable, version 2.4.1 is here to address minor bug fixes that we summarize in the following lines.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Changing the license Spark property key to be "jsl" instead of "sparkjsl" as the latter generates inconsistencies
* Fix the alignment logic for tokens and chunks in the ChunkEntityResolverSelector because when tokens and chunks did not have the same begin-end indexes the resolution was not executed

</div><div class="h3-box" markdown="1">

### 2.4.0

#### Overview

We are glad to announce Spark NLP for Healthcare 2.4.0. This is an important release because of several refactorizations achieved in the core library, plus the introduction of several state of the art algorithms, new features and enhancements.
We have included several architecture and performance improvements, that aim towards making the library more robust in terms of storage handling for Big Data.
In the NLP aspect, we have introduced a ContextualParser, DocumentLogRegClassifier and a ChunkEntityResolverSelector.
These last two Annotators also target performance time and memory consumption by lowering the order of computation and data loaded to memory in each step when designed following a hierarchical pattern.
We have put a big effort on this one, so please enjoy and share your comments. Your words are always welcome through all our different channels.
Thank you very much for your important doubts, bug reports and feedback; they are always welcome and much appreciated.

</div><div class="h3-box" markdown="1">

#### New Features

* BigChunkEntityResolver Annotator: New experimental approach to reduce memory consumption at expense of disk IO.
* ContextualParser Annotator: New entity parser that works based on context parameters defined in a JSON file.
* ChunkEntityResolverSelector Annotator: New AnnotatorModel that takes advantage of the RecursivePipelineModel + LazyAnnotator pattern to annotate with different LazyAnnotators at runtime.
* DocumentLogregClassifier Annotator: New Annotator that provides a wrapped TFIDF Vectorizer + LogReg Classifier for TOKEN AnnotatorTypes (either at Document level or Chunk level)

</div><div class="h3-box" markdown="1">

#### Enhancements

* `normalizedColumn` Param is no longer required in ChunkEntityResolver Annotator (defaults to the `labelCol` Param value).
* ChunkEntityResolverMetadata now has more data to infer whether the match is meaningful or not.

</div><div class="h3-box" markdown="1">

#### Bugfixes

* Fixed a bug on ContextSpellChecker Annotator where unrecognized tokens would cause an exception if not in vocabulary.
* Fixed a bug on ChunkEntityResolver Annotator where undetermined results were coming out of negligible confidence scores for matches.
* Fixed a bug on ChunkEntityResolver Annotator where search would fail if the `neighbours` Param was grater than the number of nodes in the tree. Now it returns up to the number of nodes in the tree.

</div><div class="h3-box" markdown="1">

#### Deprecations

* OCR Moves to its own JSL Spark OCR project.

</div>

#### Infrastructure

* Spark NLP License is now required to utilize the library. Please follow the instructions on the shared email.
