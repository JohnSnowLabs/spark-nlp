---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.5.0
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_5_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.5.0
We are glad to announce that Spark NLP Healthcare 3.5.0 has been released!

#### Highlights
+ **Zero-shot Relation Extraction** to extract relations between clinical entities with no training dataset
+ **Deidentification**:
  - New **French** **Deidentification** NER models and pipeline
  - New **Italian** **Deidentification** NER models and pipeline
  - Check our reference table for **French and Italian deidentification metrics**
  - Added **French support to the "fake" generation of data** (aka data obfuscation) in the Deidentification annotator
  - **Deidentification** **benchmark**: Spark NLP vs Cloud Providers (AWS, Azure, GCP)
+ **Graph generation**:
  - **ChunkMapperApproach** to augment NER chunks extracted by Spark NLP with a custom **graph-like dictionary of relationships**
+ **New Relation Extraction features**:
  - Configuration of **case sensitivity** in the name of the **relations** in **Relation Extraction Models**
+ **Models and Demos**:
  - We have reached **600 clinical models and pipelines**, what sums up to **5000+ overall models** in [Models Hub](https://nlp.johnsnowlabs.com/models)!
  - Check our new [live demos](https://nlp.johnsnowlabs.com/demos) including [multilanguage deidentification](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT_MULTI/) to anonymize clinical notes in 5 different languages
+ Generate Dataframes to **train Assertion Status models** using **JSON Files** exported **from Annotation Lab** (ALAB)
+ Guide about how to scale **from PoC to Production** using Spark NLP for Healthcare in our new Medium Article, available [here](https://medium.com/spark-nlp/deploying-spark-nlp-for-healthcare-from-zero-to-hero-88949b0c866d)
+ **Core improvements**:
  - **Contextual Parser** (our Rule-based NER annotator) is now **much more performant**!
  - **Bug fixing and compatibility additions** affecting and improving some behaviours of _AssertionDL, BertSentenceChunkEmbeddings, AssertionFilterer and EntityRulerApproach_
+ **New notebooks: zero-shot relation extraction and Deidentification benchmark vs Cloud Providers**

#### Zero-shot Relation Extraction to extract relations between clinical entities with no training dataset
This release includes a zero-shot relation extraction model that leverages `BertForSequenceClassificaiton` to return, based on a predefined set of relation candidates (including no-relation / O), which one has the higher probability to be linking two entities.

The dataset will be a csv which contains the following columns: `sentence`, `chunk1`, `firstCharEnt1`, `lastCharEnt1`, `label1`, `chunk2`, `firstCharEnt2`, `lastCharEnt2`, `label2`, `rel`.

For example, let's take a look at this dataset (columns `chunk1`, `rel`, `chunk2` and `sentence`):

```
+----------------------------------------------+-------+-------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| chunk1                                       | rel   | chunk2                              | sentence                                                                                                                                                                       |
|----------------------------------------------+-------+-------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| light-headedness                             | PIP   | diaphoresis                         | She states this light-headedness is often associated with shortness of breath and diaphoresis occasionally with nausea .                                                       |
| respiratory rate                             | O     | saturation                          | VITAL SIGNS - Temp 98.8 , pulse 60 , BP 150/94 , respiratory rate 18 , and saturation 96% on room air .                                                                        |
| lotions                                      | TrNAP | incisions                           | No lotions , creams or powders to incisions .                                                                                                                                  |
| abdominal ultrasound                         | TeRP  | gallbladder sludge                  | Abdominal ultrasound on 2/23/00 - This study revealed gallbladder sludge but no cholelithiasis .                                                                               |
| ir placement of a drainage catheter          | TrAP  | his abdominopelvic fluid collection | At that time he was made NPO with IVF , placed on Ampicillin / Levofloxacin / Flagyl and underwent IR placement of a drainage catheter for his abdominopelvic fluid collection |
+----------------------------------------------+-------+-------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

The relation types (TeRP, TrAP, PIP, TrNAP, etc...) are described [here](https://www.i2b2.org/NLP/Relations/assets/Relation%20Annotation%20Guideline.pdf)

Let's take a look at the first sentence!

`She states this light-headedness is often associated with shortness of breath and diaphoresis occasionally with nausea`

As we see in the table, the sentences includes a `PIP` relationship (`Medical problem indicates medical problem`), meaning that in that sentence, chunk1 (`light-headedness`) *indicates* chunk2 (`diaphoresis`).

We set a list of candidates tags (`[PIP, TrAP, TrNAP, TrWP, O]`) and candidate sentences (`[light-headedness caused diaphoresis, light-headedness was administered for diaphoresis, light-headedness was not given for diaphoresis, light-headedness worsened diaphoresis]`), meaning that:

- `PIP` is expressed by `light-headedness caused diaphoresis`
- `TrAP` is expressed by `light-headedness was administered for diaphoresis`
- `TrNAP` is expressed by `light-headedness was not given for diaphoresis`
- `TrWP` is expressed by `light-headedness worsened diaphoresis`
- or something generic, like `O` is expressed by `light-headedness and diaphoresis`...

We will get that the biggest probability of is `PIP`, since it's phrase `light-headedness caused diaphoresis` is the most similar relationship expressing the meaning in the original sentence (`light-headnedness is often associated with ... and diaphoresis`)

The example code is the following:
```
...
re_ner_chunk_filter = sparknlp_jsl.annotator.RENerChunksFilter() \
    .setRelationPairs(["problem-test","problem-treatment"]) \
    .setMaxSyntacticDistance(4)\
    .setDocLevelRelations(False)\
    .setInputCols(["ner_chunks", "dependencies"]) \
    .setOutputCol("re_ner_chunks")

# The relations are defined by a map- keys are relation label, values are lists of predicated statements. The variables in curly brackets are NER entities, there could be more than one, e.g. "{{TREATMENT, DRUG}} improves {{PROBLEM}}"
re_model = sparknlp_jsl.annotator.ZeroShotRelationExtractionModel \
    .pretrained("re_zeroshot_biobert", "en", "clinical/models")\
    .setRelationalCategories({
        "CURE": ["{{TREATMENT}} cures {{PROBLEM}}."],
        "IMPROVE": ["{{TREATMENT}} improves {{PROBLEM}}.", "{{TREATMENT}} cures {{PROBLEM}}."],
        "REVEAL": ["{{TEST}} reveals {{PROBLEM}}."]})\
    .setMultiLabel(False)\
    .setInputCols(["re_ner_chunks", "sentences"]) \
    .setOutputCol("relations")

pipeline = sparknlp.base.Pipeline() \
    .setStages([documenter, tokenizer, sentencer, words_embedder, pos_tagger, ner_tagger, ner_converter,
                dependency_parser, re_ner_chunk_filter, re_model])

data = spark.createDataFrame(
    [["Paracetamol can alleviate headache or sickness. An MRI test can be used to find cancer."]]
).toDF("text")

model = pipeline.fit(data)
results = model.transform(data)

results\
    .selectExpr("explode(relations) as relation")\
    .show(truncate=False)    
```

Results:
```
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|relation                                                                                                                                                                                                                                                                                                                                                              |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|{category, 534, 613, REVEAL, {entity1_begin -> 48, relation -> REVEAL, hypothesis -> An MRI test reveals cancer., confidence -> 0.9760039, nli_prediction -> entail, entity1 -> TEST, syntactic_distance -> 4, chunk2 -> cancer, entity2_end -> 85, entity1_end -> 58, entity2_begin -> 80, entity2 -> PROBLEM, chunk1 -> An MRI test, sentence -> 1}, []}            |
|{category, 267, 357, IMPROVE, {entity1_begin -> 0, relation -> IMPROVE, hypothesis -> Paracetamol improves sickness., confidence -> 0.98819494, nli_prediction -> entail, entity1 -> TREATMENT, syntactic_distance -> 3, chunk2 -> sickness, entity2_end -> 45, entity1_end -> 10, entity2_begin -> 38, entity2 -> PROBLEM, chunk1 -> Paracetamol, sentence -> 0}, []}|
|{category, 0, 90, IMPROVE, {entity1_begin -> 0, relation -> IMPROVE, hypothesis -> Paracetamol improves headache., confidence -> 0.9929625, nli_prediction -> entail, entity1 -> TREATMENT, syntactic_distance -> 2, chunk2 -> headache, entity2_end -> 33, entity1_end -> 10, entity2_begin -> 26, entity2 -> PROBLEM, chunk1 -> Paracetamol, sentence -> 0}, []}    |
+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

Take a look at the example notebook [here](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.3.ZeroShot_Clinical_Relation_Extraction.ipynb).

Stay tuned for the **few-shot** Annotator to be release soon!


#### New French Deidentification NER models and pipeline
We trained two new NER models to find PHI data (protected health information) that may need to be deidentified in **French**. `ner_deid_generic` and `ner_deid_subentity` models are trained with in-house annotations.
+ `ner_deid_generic` : Detects 7 PHI entities in French (`DATE`, `NAME`, `LOCATION`, `PROFESSION`, `CONTACT`, `AGE`, `ID`).
+ `ner_deid_subentity` : Detects 15 PHI sub-entities in French (`PATIENT`, `HOSPITAL`, `DATE`, `ORGANIZATION`, `E-MAIL`, `USERNAME`, `ZIP`, `MEDICALRECORD`, `PROFESSION`, `PHONE`, `DOCTOR`, `AGE`, `STREET`, `CITY`, `COUNTRY`).
*Example* :
```bash
...
embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "fr")\
    .setInputCols(["sentence", "token"])\
	   .setOutputCol("embeddings")
deid_ner = MedicalNerModel.pretrained("ner_deid_generic", "fr", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")
deid_sub_entity_ner = MedicalNerModel.pretrained("ner_deid_subentity", "fr", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_sub_entity")
...
text = """J'ai vu en consultation Michel Martinez (49 ans) adressé au Centre Hospitalier De Plaisir pour un diabète mal contrôlé avec des symptômes datant de Mars 2015."""
result = model.transform(spark.createDataFrame([[text]], ["text"]))
```
*Results* :

```bash
| chunk              		| ner_deid_generic_chunk | ner_deid_subentity_chunk |
|-------------------------------|------------------------|--------------------------|
| Michel Martinez    		| NAME                   | PATIENT                  |
| 49 ans             		| AGE                    | AGE                      |
| Centre Hospitalier De Plaisir | LOCATION        	 | HOSPITAL                 |
| Mars 2015          		| DATE                   | DATE                     |
```

We also developed a clinical deidentification pretrained pipeline that can be used to deidentify PHI information from **French** medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate the following entities: `DATE`, `AGE`, `SEX`, `PROFESSION`, `ORGANIZATION`, `PHONE`, `E-MAIL`, `ZIP`, `STREET`, `CITY`, `COUNTRY`, `PATIENT`, `DOCTOR`, `HOSPITAL`, `MEDICALRECORD`, `SSN`, `IDNUM`, `ACCOUNT`, `PLATE`, `USERNAME`, `URL`, and `IPADDR`.

```bash
from sparknlp.pretrained import PretrainedPipeline
deid_pipeline = PretrainedPipeline("clinical_deidentification", "fr", "clinical/models")
text = """PRENOM : Jean NOM : Dubois NUMÉRO DE SÉCURITÉ SOCIALE : 1780160471058 ADRESSE : 18 Avenue Matabiau VILLE : Grenoble CODE POSTAL : 38000"""
result = deid_pipeline.annotate(text)
```
*Results*:

```bash
Masked with entity labels
------------------------------
PRENOM : <PATIENT> NOM : <PATIENT> NUMÉRO DE SÉCURITÉ SOCIALE : <SSN>  ADRESSE : <STREET> VILLE : <CITY> CODE POSTAL : <ZIP>
Masked with chars
------------------------------
PRENOM : [**] NOM : [****] NUMÉRO DE SÉCURITÉ SOCIALE : [***********]  ADRESSE : [****************] VILLE : [******] CODE POSTAL : [***]
Masked with fixed length chars
------------------------------
PRENOM : **** NOM : **** NUMÉRO DE SÉCURITÉ SOCIALE : ****  ADRESSE : **** VILLE : **** CODE POSTAL : ****
Obfuscated
------------------------------
PRENOM : Mme Olivier NOM : Mme Traore NUMÉRO DE SÉCURITÉ SOCIALE : 164033818514436  ADRESSE : 731, boulevard de Legrand VILLE : Sainte Antoine CODE POSTAL : 37443
```


#### New Italian Deidentification NER models and pipeline

We trained two new NER models to find PHI data (protected health information) that may need to be deidentified in **Italian**. `ner_deid_generic` and `ner_deid_subentity` models are trained with in-house annotations.
+ `ner_deid_generic` : Detects 8 PHI entities in Italian (`DATE`, `NAME`, `LOCATION`, `PROFESSION`, `CONTACT`, `AGE`, `ID`, `SEX`).
+ `ner_deid_subentity` : Detects 19 PHI sub-entities in Italian (`DATE`, `AGE`, `SEX`, `PROFESSION`, `ORGANIZATION`, `PHONE`, `EMAIL`, `ZIP`, `STREET`, `CITY`, `COUNTRY`, `PATIENT`, `DOCTOR`, `HOSPITAL`, `MEDICALRECORD`, `SSN`, `IDNUM`, `USERNAME`, `URL`).
*Example* :
```bash
...
embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "it")\
    .setInputCols(["sentence", "token"])\
	   .setOutputCol("embeddings")
deid_ner = MedicalNerModel.pretrained("ner_deid_generic", "it", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")
deid_sub_entity_ner = MedicalNerModel.pretrained("ner_deid_subentity", "it", "clinical/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner_sub_entity")
...
text = """Ho visto Gastone Montanariello (49 anni) riferito all' Ospedale San Camillo per diabete mal controllato con sintomi risalenti a marzo 2015."""
result = model.transform(spark.createDataFrame([[text]], ["text"]))
```
*Results* :

```bash
| chunk                | ner_deid_generic_chunk | ner_deid_subentity_chunk |
|----------------------|------------------------|--------------------------|
| Gastone Montanariello| NAME                   | PATIENT                  |
| 49                   | AGE                    | AGE                      |
| Ospedale San Camillo | LOCATION               | HOSPITAL                 |
| marzo 2015           | DATE                   | DATE                     |
```

We also developed a clinical deidentification pretrained pipeline that can be used to deidentify PHI information from **Italian** medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate the following entities: `DATE`, `AGE`, `SEX`, `PROFESSION`, `ORGANIZATION`, `PHONE`, `E-MAIL`, `ZIP`, `STREET`, `CITY`, `COUNTRY`, `PATIENT`, `DOCTOR`, `HOSPITAL`, `MEDICALRECORD`, `SSN`, `IDNUM`, `ACCOUNT`, `PLATE`, `USERNAME`, `URL`, and `IPADDR`.

```bash
from sparknlp.pretrained import PretrainedPipeline
deid_pipeline = PretrainedPipeline("clinical_deidentification", "it", "clinical/models")
sample_text = """NOME: Stefano Montanariello CODICE FISCALE: YXYGXN51C61Y662I INDIRIZZO: Viale Burcardo 7 CODICE POSTALE: 80139"""
result = deid_pipeline.annotate(sample_text)
```
*Results*:

```bash
Masked with entity labels
------------------------------
NOME: <PATIENT> CODICE FISCALE: <SSN> INDIRIZZO: <STREET> CODICE POSTALE: <ZIP>

Masked with chars
------------------------------
NOME: [*******************] CODICE FISCALE: [**************] INDIRIZZO: [**************] CODICE POSTALE: [***]

Masked with fixed length chars
------------------------------
NOME: **** CODICE FISCALE: **** INDIRIZZO: **** CODICE POSTALE: ****

Obfuscated
------------------------------
NOME: Stefania Gregori CODICE FISCALE: UIWSUS86M04J604B INDIRIZZO: Viale Orlando 808 CODICE POSTALE: 53581
```

#### Check our reference table for **French and Italian deidentification metrics**
Please find this reference table with metrics comparing F1 score for the available entities in French and Italian clinical pipelines:
```
|Entity Label |Italian|French|
|-------------|-------|------|
|PATIENT      |0.9069 |0.9382|
|DOCTOR       |0.9171 |0.9912|
|HOSPITAL     |0.8968 |0.9375|
|DATE         |0.9835 |0.9849|
|AGE          |0.9832 |0.8575|
|PROFESSION   |0.8864 |0.8147|
|ORGANIZATION |0.7385 |0.7697|
|STREET       |0.9754 |0.8986|
|CITY         |0.9678 |0.8643|
|COUNTRY      |0.9262 |0.8983|
|PHONE        |0.9815 |0.9785|
|USERNAME     |0.9091 |0.9239|
|ZIP          |0.9867 |1.0   |
|E-MAIL       |1      |1.0   |
|MEDICALRECORD|0.8085 |0.939 |
|SSN          |0.9286 |N/A   |
|URL          |1      |N/A   |
|SEX          |0.9697 |N/A   |
|IDNUM        |0.9576 |N/A   |
```



#### Added French support in Deidentification Annotator for data obfuscation
Our `Deidentificator` annotator is now able to obfuscate entities (coming from a deid NER model) with fake data in French language. Example:

Example code:
```
...
embeddings = WordEmbeddingsModel.pretrained("w2v_cc_300d", "fr").setInputCols(["sentence", "token"]).setOutputCol("word_embeddings")

clinical_ner = MedicalNerModel.pretrained("ner_deid_subentity", "fr", "clinical/models").setInputCols(["sentence","token", "word_embeddings"]).setOutputCol("ner")

ner_converter = NerConverter().setInputCols(["sentence", "token", "ner"]).setOutputCol("ner_chunk")

de_identification = DeIdentification() \
    .setInputCols(["ner_chunk", "token", "sentence"]) \
    .setOutputCol("dei") \
    .setMode("obfuscate") \
    .setObfuscateDate(True) \
    .setRefSep("#") \
    .setDateTag("DATE") \
    .setLanguage("fr") \
    .setObfuscateRefSource('faker')

pipeline = Pipeline() \
    .setStages([
    documentAssembler,
    sentenceDetector,
    tokenizer,
    embeddings,
    clinical_ner,
    ner_converter,
    de_identification
])
sentences = [
["""J'ai vu en consultation Michel Martinez (49 ans) adressé au Centre Hospitalier De Plaisir pour un diabète mal contrôlé avec des symptômes datant"""]
]

my_input_df = spark.createDataFrame(sentences).toDF("text")
output = pipeline.fit(my_input_df).transform(my_input_df)
...
```

Entities detected:
```
+------------+----------+
|token       |entity    |
+------------+----------+
|J'ai        |O         |
|vu          |O         |
|en          |O         |
|consultation|O         |
|Michel      |B-PATIENT |
|Martinez    |I-PATIENT |
|(           |O         |
|49          |B-AGE     |
|ans         |O         |
|)           |O         |
|adressé     |O         |
|au          |O         |
|Centre      |B-HOSPITAL|
|Hospitalier |I-HOSPITAL|
|De          |I-HOSPITAL|
|Plaisir     |I-HOSPITAL|
|pour        |O         |
|un          |O         |
|diabète     |O         |
|mal         |O         |
+------------+----------+
```

Obfuscated sentence:
```
+--------------------------------------------------------------------------------------------------------------------------------------------------------+
|result                                                                                                                                                  |
+--------------------------------------------------------------------------------------------------------------------------------------------------------+
|[J'ai vu en consultation Sacrispeyre Ligniez (86 ans) adressé au Centre Hospitalier Pierre Futin pour un diabète mal contrôlé avec des symptômes datant]|
+--------------------------------------------------------------------------------------------------------------------------------------------------------+
```


#### Deidentification benchmark: Spark NLP vs Cloud Providers (AWS, Azure, GCP)
We have published a new notebook with a benchmark and the reproduceable code, comparing Spark NLP for Healthcare Deidentification capabilities of one of our English pipelines (`clinical_deidentification_glove_augmented`) versus:
- AWS Comprehend Medical
- Azure Cognitive Services
- GCP Data Loss Prevention

The notebook is available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.3.Clinical_Deidentification_SparkNLP_vs_Cloud_Providers_Comparison.ipynb), and the results are the following:

```
        SPARK NLP   AWS    AZURE      GCP
AGE          1      0.96    0.93      0.9
DATE         1      0.99    0.9       0.96
DOCTOR      0.98    0.96    0.7       0.6
HOSPITAL    0.92    0.89    0.72      0.72
LOCATION    0.9     0.81    0.87      0.73
PATIENT     0.96    0.95    0.78      0.48
PHONE        1       1      0.8       0.97
ID          0.93    0.93     -          -
```

#### ChunkMapperApproach: mapping extracted entities to an ontology (Json dictionary) with relations
We have released a new annotator, called **ChunkMapperApproach**(), that receives a **ner_chunk** and a Json with a mapping of NER entities and relations, and returns the **ner_chunk** augmented with the relations from the Json ontology.


Example of a small ontology with relations:


Giving the map with entities and relationships stored in mapper.json, we will use an NER to detect entities in a text and, in case any of them is found, the **ChunkMapper** will augment the output with the relationships from this dictionary:

```
{"mappings": [{
             "key": "metformin",
             "relations": [{
                   "key": "action",
                   "values" : ["hypoglycemic", "Drugs Used In Diabets"]
                   },{
                   "key": "treatment",
                   "values" : ["diabetes", "t2dm"]
                   }]
           }]
```

```
text = ["""The patient was prescribed 1 unit of Advil for 5 days after meals. The patient was also
given 1 unit of Metformin daily.
He was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night ,
12 units of insulin lispro with meals , and metformin 1000 mg two times a day."""]
...
nerconverter = NerConverterInternal()\
  .setInputCols("sentence", "token", "ner")\
  .setOutputCol("ner_chunk")

chunkerMapper = ChunkMapperApproach() \
  .setInputCols("ner_chunk")\
  .setOutputCol("relations")\
  .setDictionary("mapper.json")\
  .setRel("action")

pipeline = Pipeline().setStages([document_assembler,sentence_detector,tokenizer, ner, nerconverter, chunkerMapper])

res = pipeline.fit(test_data).transform(test_data)

res.select(F.explode('ner_chunk.result').alias("chunks")).show(truncate=False)
```

Entities:
```
+----------------+
|chunks          |
+----------------+
|Metformin       |
|insulin glargine|
|insulin lispro  |
|metformin       |
|mg              |
|times           |
+----------------+
```

Checking the relations:
```
...
pd_df = res.select(F.explode('relations').alias('res')).select('res.result', 'res.metadata').toPandas()
...
```

Results:
```
Entity:					metformin
Main relation:				hypoglycemic
Other relations (included in metadata):	Drugs Used In Diabets
```


#### Configuration of case sensitivity in the name of the relations in Relation Extraction Models
We have added a new parameter, called 'relationPairsCaseSensitive', which affects the way `setRelationPairs` works. If `relationPairsCaseSensitive` is True, then the pairs of entities in the dataset should match the pairs in setRelationPairs in their specific case (case sensitive). By default it's set to False, meaning that the match of those relation names is case insensitive.

Before 3.5.0, `.setRelationPairs(["dosage-drug"])` would not return relations if it was trained with a relation called `DOSAGE-DRUG` (different casing). Now, setting `.setRelationPairs(["dosage-drug"])`and `relationPairsCaseSensitive(False)` or just leaving it by default, it will return any `dosage-drug` or `DOSAGE-DRUG` relationship.

Example of usage in Python:
```
...
reModel = RelationExtractionModel()\
    .pretrained("posology_re")\
    .setInputCols(["embeddings", "pos_tags", "ner_chunks", "dependencies"])\
    .setMaxSyntacticDistance(4)\
    .setRelationPairs(["dosage-drug"]) \
    .setRelationPairsCaseSensitive(False) \
    .setOutputCol("relations_case_insensitive")
...
```

This will return relations named dosage-drug, DOSAGE-DRUG, etc.


#### We have reached the milestone of 600 clinical models (and 5000+ models overall) ! 🥳
This release added to Spark NLP Models Hub 100+ pretrained clinical pipelines, available to use as one-liners, including some of the most used NER models, namely:

+ `ner_deid_generic_pipeline_de`: German deidentification pipeline with aggregated (generic) labels
+ `ner_deid_subentity_pipeline_de`: German deidentification pipeline with specific (subentity) labels
+ `ner_clinical_biobert_pipeline_en`: A pretrained pipeline based on `ner_clinical_biobert` to carry out NER on BioBERT embeddings
+ `ner_abbreviation_clinical_pipeline_en`: A pretrained pipeline based on `ner_abbreviation_clinical` that detects medical acronyms and abbreviations
+ `ner_ade_biobert_pipeline_en`: A pretrained pipeline based on `ner_ade_biobert` to carry out Adverse Drug Events NER recognition using BioBERT embeddings
+ `ner_ade_clinical_pipeline_en`: Similar to the previous one, but using `clinical_embeddings`
+ `ner_radiology_pipeline_en`: A pretrained pipeline to detect Radiology entities (coming from `ner_radiology_wip` model)
+ `ner_events_clinical_pipeline_en`: A pretrained pipeline to extract Clinical Events related entities (leveraging `ner_events_clinical`)
+ `ner_anatomy_biobert_pipeline_en`: A pretrained pipeline to extract Anamoty entities (from `ner_anamoty_biobert`)
+ ...100 more

Here is how you can use any of the pipelines with one line of code:

```
from sparknlp.pretrained import PretrainedPipeline

pipeline = PretrainedPipeline("explain_clinical_doc_medication", "en", "clinical/models")

result = pipeline.fullAnnotate("""The patient is a 30-year-old female with a long history of insulin dependent diabetes, type 2. She received a course of Bactrim for 14 days for UTI.  She was prescribed 5000 units of Fragmin  subcutaneously daily, and along with Lantus 40 units subcutaneously at bedtime.""")[0]
```

Results:
```
+----+----------------+------------+
|    | chunks         | entities   |
|---:|:---------------|:-----------|
|  0 | insulin        | DRUG       |
|  1 | Bactrim        | DRUG       |
|  2 | for 14 days    | DURATION   |
|  3 | 5000 units     | DOSAGE     |
|  4 | Fragmin        | DRUG       |
|  5 | subcutaneously | ROUTE      |
|  6 | daily          | FREQUENCY  |
|  7 | Lantus         | DRUG       |
|  8 | 40 units       | DOSAGE     |
|  9 | subcutaneously | ROUTE      |
| 10 | at bedtime     | FREQUENCY  |
+----+----------------+------------+
+----+----------+------------+-------------+
|    | chunks   | entities   | assertion   |
|---:|:---------|:-----------|:------------|
|  0 | insulin  | DRUG       | Present     |
|  1 | Bactrim  | DRUG       | Past        |
|  2 | Fragmin  | DRUG       | Planned     |
|  3 | Lantus   | DRUG       | Planned     |
+----+----------+------------+-------------+
+----------------+-----------+------------+-----------+----------------+
| relation       | entity1   | chunk1     | entity2   | chunk2         |
|:---------------|:----------|:-----------|:----------|:---------------|
| DRUG-DURATION  | DRUG      | Bactrim    | DURATION  | for 14 days    |
| DOSAGE-DRUG    | DOSAGE    | 5000 units | DRUG      | Fragmin        |
| DRUG-ROUTE     | DRUG      | Fragmin    | ROUTE     | subcutaneously |
| DRUG-FREQUENCY | DRUG      | Fragmin    | FREQUENCY | daily          |
| DRUG-DOSAGE    | DRUG      | Lantus     | DOSAGE    | 40 units       |
| DRUG-ROUTE     | DRUG      | Lantus     | ROUTE     | subcutaneously |
| DRUG-FREQUENCY | DRUG      | Lantus     | FREQUENCY | at bedtime     |
+----------------+-----------+------------+-----------+----------------+
```

We have updated our [11.Pretrained_Clinical_Pipelines.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/11.Pretrained_Clinical_Pipelines.ipynb) notebook to properly show this addition. Don't forget to check it out!

All of our scalable, production-ready Spark NLP Clinical Models and Pipelines can be found in our Models Hub

Finally, we have added two new **entityMapper** models: **drug_ontology** and **section_mapper**

For all Spark NLP for healthcare models, please check our [Models Hub webpage](https://nlp.johnsnowlabs.com/models?edition=Spark+NLP+for+Healthcare)


#### Have you checked our demo page?
New several demos were created, available at https://nlp.johnsnowlabs.com/demos

In this release we feature the **Multilingual deidentification**, showcasing how to deidentify clinical texts in English, Spanish, German, French and Italian. This demo is available [here](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT_MULTI)

**For the rest of the demos, please visit [Models Hub Demos Page](https://nlp.johnsnowlabs.com/demos)**

#### Generate Dataframes to train Assertion Status Models using JSON files exported from Annotation Lab (ALAB)
Now we can generate a dataframe that can be used to train an `AssertionDLModel` by using the output of `AnnotationToolJsonReader.generatePlainAssertionTrainSet()`. The dataframe contains all the columns that you need for training.

*Example* :

```python
filename = "../json_import.json"
reader = AnnotationToolJsonReader(assertion_labels = ['AsPresent', 'AsAbsent', 'AsConditional', 'AsHypothetical', 'AsFamily', 'AsPossible', 'AsElse'])
df =  reader.readDataset(spark, filename)
reader.generatePlainAssertionTrainSet(df).show(truncate=False)
```

*Results* :

```
+-------+--------------------------------------------+-----+---+-----------+---------+
|task_id|sentence                                    |begin|end|ner        |assertion|
+-------+--------------------------------------------+-----+---+-----------+---------+
|1      |Patient has a headache for the last 2 weeks |2    |3  |a headache |AsPresent|
+-------+--------------------------------------------+-----+---+-----------+---------+
```

#### Understand how to scale from a PoC to Production using Spark NLP for Healthcare in our new Medium Article, available here

We receive many questions about how Spark work distribution is carried out, what specially becomes important before making the leap from a PoC to a big scalable, production-ready cluster.

This article helps you understand:
- How many different ways to create a cluster are available, as well as their advantages and disadvantages;
- How to scale all of them;
- How to take advantage of autoscalability and autotermination policy in Cloud Providers;
- Which are the steps to take depending on your infrastructure, to make the leap to production;

If you need further assistance, please reach our Support team at [support@johnsnowlabs.com](mailto:support@johnsnowlabs.com)


#### Contextual Parser (our Rule-based NER annotator) is now much more performant!
Contextual Parser has been improved in terms of performance. These are the metrics comparing 3.4.2 and 3.5.0

```
4 cores and 30 GB RAM
=====================
	10 MB	20 MB	30MB	50MB		
3.4.2	349	786	982	1633		
3.5.0   142	243	352	556		

8 cores and 60 GB RAM
=====================
	10 MB	20 MB	30MB	50MB
3.4.2	197	373	554	876
3.5.0   79	136	197	294
```

#### We have reached the milestone of 600 clinical demos!
During this release, we included:
- More than 100+ recently created clinical models and pipelines, including NER, NER+RE, NER+Assertion+RE, etc.
- Added two new `entityMapper` models: `drug_action_treatment_mapper` and `normalized_section_header_mapper`

**For all Spark NLP for healthcare models, please check : [Models Hub Page](https://nlp.johnsnowlabs.com/models?edition=Spark+NLP+for+Healthcare)**



#### Bug fixing and compatibility additions
This is the list of fixed issues and bugs, as well as one compatibility addition between **EntityRuler** and **AssertionFiltered**:

+ **Error in AssertionDLApproach and AssertionLogRegApproach**: an error was being triggered wthen the dataset contained long (64bits) instead of 32 bits integers for the start / end columns. Now this bug is fixed.
+ **Error in BertSentenceChunkEmbeddings**: loading a model after downloading it with pretrained() was triggering an error. Now you can load any model after downloading it with `pretrained()`.
+ Adding **setIncludeConfidence** to AssertionDL Python version, where it was missing. Now, it's included in both Python and Scala, as described [here](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/assertion/dl/AssertionDLModel.html#setIncludeConfidence(value:Boolean):AssertionDLModel.this.type)
+ **Making EntityRuler and AssertionFiltered compatible**: AssertionFilterer annotator that is being used to filter the entities based on entity labels now can be used by EntityRulerApproach, a rule based entity extractor:

```
Path("test_file.jsonl").write_text(json.dumps({"id":"cough","label":"COUGH","patterns":["cough","coughing"]}))
...
entityRuler = EntityRulerApproach()\
    .setInputCols(["sentence", "token"])\
    .setOutputCol("ner_chunk")\
    .setPatternsResource("test_file.jsonl", ReadAs.TEXT, {"format": "jsonl"})

clinical_assertion = AssertionDLModel.pretrained("assertion_dl", "en", "clinical/models") \
    .setInputCols(["sentence", "ner_chunk", "embeddings"]) \
    .setOutputCol("assertion")

assertion_filterer = AssertionFilterer()\
    .setInputCols("sentence","ner_chunk","assertion")\
    .setOutputCol("assertion_filtered")\
    .setWhiteList(["present"])\

...

empty_data = spark.createDataFrame([[""]]).toDF("text")
ruler_model = rulerPipeline.fit(empty_data)

text = "I have a cough but no fatigue or chills."

ruler_light_model = LightPipeline(ruler_model).fullAnnotate(text)[0]['assertion_filtered']
```

Result:
```
Annotation(chunk, 9, 13, cough, {'entity': 'COUGH', 'id': 'cough', 'sentence': '0'})]
```


#### **New notebooks: zero-shot relation extraction and Deidentification benchmark (Spark NLP and Cloud Providers)**
Check these recently notebooks created by our Healthcare team and available in our [Spark NLP Workshop git repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/), where you can find many more.
- Zero-shot Relation Extraction, available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.3.ZeroShot_Clinical_Relation_Extraction.ipynb).
- Deidentification benchmark (SparkNLP and Cloud Providers), available [here](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/4.3.Clinical_Deidentification_SparkNLP_vs_Cloud_Providers_Comparison.ipynb)

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_4_2">Version 3.4.2</a>
    </li>
    <li>
        <strong>Version 3.5.0</strong>
    </li>
    <li>
        <a href="release_notes_3_5_1">Version 3.5.1</a>
    </li>
</ul>

<ul class="pagination owl-carousel pagination_big">
    <li><a href="release_notes_4_2_1">4.2.1</a></li>
    <li><a href="release_notes_4_2_0">4.2.0</a></li>
    <li><a href="release_notes_4_1_0">4.1.0</a></li>
    <li><a href="release_notes_4_0_2">4.0.2</a></li>
    <li><a href="release_notes_4_0_0">4.0.0</a></li>
    <li><a href="release_notes_3_5_3">3.5.3</a></li>
    <li><a href="release_notes_3_5_2">3.5.2</a></li>
    <li><a href="release_notes_3_5_1">3.5.1</a></li>
    <li class="active"><a href="release_notes_3_5_0">3.5.0</a></li>
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