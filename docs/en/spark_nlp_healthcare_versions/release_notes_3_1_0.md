---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.1.0
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_1_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.1.0
We are glad to announce that Spark NLP for Healthcare 3.1.0 has been released!
#### Highlights
+ Improved load time & memory consumption for SentenceResolver models.
+ New JSL Bert Models.
+ JSL SBert Model Speed Benchmark.
+ New ICD10CM resolver models.
+ New Deidentification NER models.
+ New column returned in DeidentificationModel
+ New Reidentification feature
+ New Deidentification Pretrained Pipelines
+ Chunk filtering based on confidence
+ Extended regex dictionary fuctionallity in Deidentification
+ Enhanced RelationExtractionDL Model to create and identify relations between entities across the entire document
+ MedicalNerApproach can now accept a graph file directly.
+ MedicalNerApproach can now accept a user-defined name for log file.
+ More improvements in Scaladocs.
+ Bug fixes in Deidentification module.
+ New notebooks.



#### Sentence Resolver Models load time improvement
Sentence resolver models now have faster load times, with a speedup of about 6X when compared to previous versions.
Also, the load process now is more  memory friendly meaning that the maximum memory required during load time is smaller, reducing the chances of OOM exceptions, and thus relaxing hardware requirements.

#### New JSL SBert Models
We trained new sBert models in TF2 and fined tuned on MedNLI, NLI and UMLS datasets with various parameters to cover common NLP tasks in medical domain. You can find the details in the following table.

+ `sbiobert_jsl_cased`
+ `sbiobert_jsl_umls_cased`
+ `sbert_jsl_medium_uncased`
+ `sbert_jsl_medium_umls_uncased`
+ `sbert_jsl_mini_uncased`
+ `sbert_jsl_mini_umls_uncased`
+ `sbert_jsl_tiny_uncased`
+ `sbert_jsl_tiny_umls_uncased`

#### JSL SBert Model Speed Benchmark

| JSL SBert Model| Base Model | Is Cased | Train Datasets | Inference speed (100 rows) |
|-|-|-|-|-|
| sbiobert_jsl_cased | biobert_v1.1_pubmed | Cased | medNLI, allNLI| 274,53 |
| sbiobert_jsl_umls_cased | biobert_v1.1_pubmed | Cased | medNLI, allNLI, umls | 274,52 |
| sbert_jsl_medium_uncased | uncased_L-8_H-512_A-8 | Uncased | medNLI, allNLI| 80,40 |
| sbert_jsl_medium_umls_uncased | uncased_L-8_H-512_A-8 | Uncased | medNLI, allNLI, umls | 78,35 |
| sbert_jsl_mini_uncased | uncased_L-4_H-256_A-4 | Uncased | medNLI, allNLI| 10,68 |
| sbert_jsl_mini_umls_uncased | uncased_L-4_H-256_A-4 | Uncased | medNLI, allNLI, umls | 10,29 |
| sbert_jsl_tiny_uncased | uncased_L-2_H-128_A-2 | Uncased | medNLI, allNLI| 4,54 |
| sbert_jsl_tiny_umls_uncased | uncased_L-2_H-128_A-2 | Uncased | medNLI, allNL, umls | 4,54 |

#### New ICD10CM resolver models:
These models map clinical entities and concepts to ICD10 CM codes using sentence bert embeddings. They also return the official resolution text within the brackets inside the metadata. Both models are augmented with synonyms, and previous augmentations are flexed according to cosine distances to unnormalized terms (ground truths).

+ `sbiobertresolve_icd10cm_slim_billable_hcc`: Trained with classic sbiobert mli. (`sbiobert_base_cased_mli`)

Models Hub Page : https://nlp.johnsnowlabs.com/2021/05/25/sbiobertresolve_icd10cm_slim_billable_hcc_en.html

+ `sbertresolve_icd10cm_slim_billable_hcc_med`: Trained with new jsl sbert(`sbert_jsl_medium_uncased`)

Models Hub Page : https://nlp.johnsnowlabs.com/2021/05/25/sbertresolve_icd10cm_slim_billable_hcc_med_en.html


*Example*: 'bladder cancer'
+ `sbiobertresolve_icd10cm_augmented_billable_hcc`

| chunks | code | all_codes | resolutions |all_distances | 100x Loop(sec) |
|-|-|-|-|-|-|
| bladder cancer | C679 | [C679, Z126, D090, D494, C7911] | [bladder cancer, suspected bladder cancer, cancer in situ of urinary bladder, tumor of bladder neck, malignant tumour of bladder neck] | [0.0000, 0.0904, 0.0978, 0.1080, 0.1281] | 26,9 |

+ ` sbiobertresolve_icd10cm_slim_billable_hcc`

| chunks | code | all_codes | resolutions |all_distances | 100x Loop(sec) |
| - | - | - | - | - | - |
| bladder cancer | D090 | [D090, D494, C7911, C680, C679] | [cancer in situ of urinary bladder [Carcinoma in situ of bladder], tumor of bladder neck [Neoplasm of unspecified behavior of bladder], malignant tumour of bladder neck [Secondary malignant neoplasm of bladder], carcinoma of urethra [Malignant neoplasm of urethra], malignant tumor of urinary bladder [Malignant neoplasm of bladder, unspecified]] | [0.0978, 0.1080, 0.1281, 0.1314, 0.1284] | 20,9 |

+ `sbertresolve_icd10cm_slim_billable_hcc_med`

| chunks | code | all_codes | resolutions |all_distances | 100x Loop(sec) |
| - | - | - | - | - | - |
| bladder cancer | C671 | [C671, C679, C61, C672, C673] | [bladder cancer, dome [Malignant neoplasm of dome of bladder], cancer of the urinary bladder [Malignant neoplasm of bladder, unspecified], prostate cancer [Malignant neoplasm of prostate], cancer of the urinary bladder] | [0.0894, 0.1051, 0.1184, 0.1180, 0.1200] | 12,8 |

#### New Deidentification NER Models

We trained four new NER models to find PHI data (protected health information) that may need to be deidentified. `ner_deid_generic_augmented` and `ner_deid_subentity_augmented` models are trained with a combination of 2014 i2b2 Deid dataset and in-house annotations as well as some augmented version of them. Compared to the same test set coming from 2014 i2b2 Deid dataset, we achieved a better accuracy and generalisation on some entity labels as summarised in the following tables. We also trained the same models with `glove_100d` embeddings to get more memory friendly versions.    

+ `ner_deid_generic_augmented`  : Detects PHI 7 entities (`DATE`, `NAME`, `LOCATION`, `PROFESSION`, `CONTACT`, `AGE`, `ID`).

Models Hub Page : https://nlp.johnsnowlabs.com/2021/06/01/ner_deid_generic_augmented_en.html

| entity| ner_deid_large (v3.0.3 and before)|ner_deid_generic_augmented (v3.1.0)|
|--|:--:|:--:|
|   CONTACT| 0.8695|0.9592 |
|      NAME| 0.9452|0.9648 |
|      DATE| 0.9778|0.9855 |
|  LOCATION| 0.8755|0.923 |

+ `ner_deid_subentity_augmented`: Detects PHI 23 entities (`MEDICALRECORD`, `ORGANIZATION`, `DOCTOR`, `USERNAME`, `PROFESSION`, `HEALTHPLAN`, `URL`, `CITY`, `DATE`, `LOCATION-OTHER`, `STATE`, `PATIENT`, `DEVICE`, `COUNTRY`, `ZIP`, `PHONE`, `HOSPITAL`, `EMAIL`, `IDNUM`, `SREET`, `BIOID`, `FAX`, `AGE`)

Models Hub Page : https://nlp.johnsnowlabs.com/2021/06/01/ner_deid_subentity_augmented_en.html

|entity| ner_deid_enriched (v3.0.3 and before)| ner_deid_subentity_augmented (v3.1.0)|
|-------------|:------:|:--------:|
|     HOSPITAL| 0.8519| 0.8983 |
|         DATE| 0.9766| 0.9854 |
|         CITY| 0.7493| 0.8075 |
|       STREET| 0.8902| 0.9772 |
|          ZIP| 0.8| 0.9504 |
|        PHONE| 0.8615| 0.9502 |
|       DOCTOR| 0.9191| 0.9347 |
|          AGE| 0.9416| 0.9469 |

+ `ner_deid_generic_glove`: Small version of `ner_deid_generic_augmented` and detects 7 entities.
+ `ner_deid_subentity_glove`: Small version of `ner_deid_subentity_augmented` and detects 23 entities.

*Example*:

Scala
```scala
...
val deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models") \
      .setInputCols(Array("sentence", "token", "embeddings")) \
      .setOutputCol("ner")
...
val nlpPipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter))
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

val result = pipeline.fit(Seq.empty["A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 -year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227."].toDS.toDF("text")).transform(data)
```

Python
```bash
...
deid_ner = MedicalNerModel.pretrained("ner_deid_subentity_augmented", "en", "clinical/models") \
      .setInputCols(["sentence", "token", "embeddings"]) \
      .setOutputCol("ner")
...
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, deid_ner, ner_converter])
model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""A. Record date : 2093-01-13, David Hale, M.D., Name : Hendrickson, Ora MR. # 7194334 Date : 01/13/93 PCP : Oliveira, 25 -year-old, Record date : 1-11-2000. Cocke County Baptist Hospital. 0295 Keats Street. Phone +1 (302) 786-5227."""]})))
```
*Results*:
```bash
+-----------------------------+-------------+
|chunk                        |ner_label    |
+-----------------------------+-------------+
|2093-01-13                   |DATE         |
|David Hale                   |DOCTOR       |
|Hendrickson, Ora             |PATIENT      |
|7194334                      |MEDICALRECORD|
|01/13/93                     |DATE         |
|Oliveira                     |DOCTOR       |
|25-year-old                  |AGE          |
|1-11-2000                    |DATE         |
|Cocke County Baptist Hospital|HOSPITAL     |
|0295 Keats Street.           |STREET       |
|(302) 786-5227               |PHONE        |
|Brothers Coal-Mine           |ORGANIZATION |
+-----------------------------+-------------+
```


#### New column returned in DeidentificationModel

DeidentificationModel now can return a new column to save the mappings between the mask/obfuscated entities and original entities.
This column is optional and you can set it up with the `.setReturnEntityMappings(True)` method. The default value is False.
Also, the name for the column can be changed using the following method; `.setMappingsColumn("newAlternativeName")`
The new column will produce annotations with the following structure,
```
Annotation(
  type: chunk,
  begin: 17,
  end: 25,
  result: 47,
    metadata:{
        originalChunk -> 01/13/93  //Original text of the chunk
        chunk -> 0  // The number of the chunk in the sentence
        beginOriginalChunk -> 95 // Start index of the original chunk
        endOriginalChunk -> 102  // End index of the original chunk
        entity -> AGE // Entity of the chunk
        sentence -> 2 // Number of the sentence
    }
)
```

#### New Reidentification feature

With the new ReidetificationModel, the user can go back to the original sentences using the mappings columns and the deidentification sentences.

*Example:*

Scala

```scala
val redeidentification = new ReIdentification()
     .setInputCols(Array("mappings", "deid_chunks"))
     .setOutputCol("original")
```

Python

```scala
reDeidentification = ReIdentification()
     .setInputCols(["mappings","deid_chunks"])
     .setOutputCol("original")
```

#### New Deidentification Pretrained Pipelines
We developed a `clinical_deidentification` pretrained pipeline that can be used to deidentify PHI information from medical texts. The PHI information will be masked and obfuscated in the resulting text. The pipeline can mask and obfuscate `AGE`, `CONTACT`, `DATE`, `ID`, `LOCATION`, `NAME`, `PROFESSION`, `CITY`, `COUNTRY`, `DOCTOR`, `HOSPITAL`, `IDNUM`, `MEDICALRECORD`, `ORGANIZATION`, `PATIENT`, `PHONE`, `PROFESSION`,  `STREET`, `USERNAME`, `ZIP`, `ACCOUNT`, `LICENSE`, `VIN`, `SSN`, `DLN`, `PLATE`, `IPADDR` entities.

Models Hub Page : [clinical_deidentification](https://nlp.johnsnowlabs.com/2021/05/27/clinical_deidentification_en.html)

There is also a lightweight version of the same pipeline trained with memory efficient `glove_100d`embeddings.
Here are the model names:
- `clinical_deidentification`
- `clinical_deidentification_glove`

*Example:*

Python:
```bash
from sparknlp.pretrained import PretrainedPipeline
deid_pipeline = PretrainedPipeline("clinical_deidentification", "en", "clinical/models")

deid_pipeline.annotate("Record date : 2093-01-13, David Hale, M.D. IP: 203.120.223.13. The driver's license no:A334455B. the SSN:324598674 and e-mail: hale@gmail.com. Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93. PCP : Oliveira, 25 years-old. Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286.")
```
Scala:
```scala
import com.johnsnowlabs.nlp.pretrained.PretrainedPipeline
val deid_pipeline = PretrainedPipeline("clinical_deidentification","en","clinical/models")

val result = deid_pipeline.annotate("Record date : 2093-01-13, David Hale, M.D. IP: 203.120.223.13. The driver's license no:A334455B. the SSN:324598674 and e-mail: hale@gmail.com. Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93. PCP : Oliveira, 25 years-old. Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286.")
```
*Result*:
```bash
{'sentence': ['Record date : 2093-01-13, David Hale, M.D.',
   'IP: 203.120.223.13.',
   'The driver's license no:A334455B.',
   'the SSN:324598674 and e-mail: hale@gmail.com.',
   'Name : Hendrickson, Ora MR. # 719435 Date : 01/13/93.',
   'PCP : Oliveira, 25 years-old.',
   'Record date : 2079-11-09, Patient's VIN : 1HGBH41JXMN109286.'],
'masked': ['Record date : <DATE>, <DOCTOR>, M.D.',
   'IP: <IPADDR>.',
   'The driver's license <DLN>.',
   'the <SSN> and e-mail: <EMAIL>.',
   'Name : <PATIENT> MR. # <MEDICALRECORD> Date : <DATE>.',
   'PCP : <DOCTOR>, <AGE> years-old.',
   'Record date : <DATE>, Patient's VIN : <VIN>.'],
'obfuscated': ['Record date : 2093-01-18, Dr Alveria Eden, M.D.',
   'IP: 001.001.001.001.',
   'The driver's license K783518004444.',
   'the SSN-400-50-8849 and e-mail: Merilynn@hotmail.com.',
   'Name : Charls Danger MR. # J3366417 Date : 01-18-1974.',
   'PCP : Dr Sina Sewer, 55 years-old.',
   'Record date : 2079-11-23, Patient's VIN : 6ffff55gggg666777.'],
'ner_chunk': ['2093-01-13',
   'David Hale',
   'no:A334455B',
   'SSN:324598674',
   'Hendrickson, Ora',
   '719435',
   '01/13/93',
   'Oliveira',
   '25',
   '2079-11-09',
   '1HGBH41JXMN109286']}
```

#### Chunk filtering based on confidence

We added a new annotator ChunkFiltererApproach that allows to load a csv with both entities and confidence thresholds.
This annotator will produce a ChunkFilterer model.

You can load the dictionary with the following property `setEntitiesConfidenceResource()`.

An example dictionary is:

```CSV
TREATMENT,0.7
```

With that dictionary, the user can filter the chunks corresponding to treatment entities which have confidence lower than 0.7.

Example:

We have a ner_chunk column and sentence column with the following data:

Ner_chunk
```
|[{chunk, 141, 163, the genomicorganization, {entity -> TREATMENT, sentence -> 0, chunk -> 0, confidence -> 0.57785}, []}, {chunk, 209, 267, a candidate gene forType II
           diabetes mellitus, {entity -> PROBLEM, sentence -> 0, chunk -> 1, confidence -> 0.6614286}, []}, {chunk, 394, 408, byapproximately, {entity -> TREATMENT, sentence -> 1, chunk -> 2, confidence -> 0.7705}, []}, {chunk, 478, 508, single nucleotide polymorphisms, {entity -> TREATMENT, sentence -> 2, chunk -> 3, confidence -> 0.7204666}, []}, {chunk, 559, 581, aVal366Ala substitution, {entity -> TREATMENT, sentence -> 2, chunk -> 4, confidence -> 0.61505}, []}, {chunk, 588, 601, an 8 base-pair, {entity -> TREATMENT, sentence -> 2, chunk -> 5, confidence -> 0.29226667}, []}, {chunk, 608, 625, insertion/deletion, {entity -> PROBLEM, sentence -> 3, chunk -> 6, confidence -> 0.9841}, []}]|
+-------
```
Sentence
```
[{document, 0, 298, The human KCNJ9 (Kir 3.3, GIRK3) is a member of the G-protein-activated inwardly rectifying potassium (GIRK) channel family.Here we describe the genomicorganization of the KCNJ9 locus on chromosome 1q21-23 as a candidate gene forType II
             diabetes mellitus in the Pima Indian population., {sentence -> 0}, []}, {document, 300, 460, The gene spansapproximately 7.6 kb and contains one noncoding and two coding exons ,separated byapproximately 2.2 and approximately 2.6 kb introns, respectively., {sentence -> 1}, []}, {document, 462, 601, We identified14 single nucleotide polymorphisms (SNPs),
             including one that predicts aVal366Ala substitution, and an 8 base-pair, {sentence -> 2}, []}, {document, 603, 626, (bp) insertion/deletion., {sentence -> 3}, []}]
```

We can filter the entities using the following annotator:

```python
        chunker_filter = ChunkFiltererApproach().setInputCols("sentence", "ner_chunk") \
            .setOutputCol("filtered") \
            .setCriteria("regex") \
            .setRegex([".*"]) \         
            .setEntitiesConfidenceResource("entities_confidence.csv")

```
Where entities-confidence.csv has the following data:

```csv
TREATMENT,0.7
PROBLEM,0.9
```
We can use that chunk_filter:

```
chunker_filter.fit(data).transform(data)
```
Producing the following entities:

```
|[{chunk, 394, 408, byapproximately, {entity -> TREATMENT, sentence -> 1, chunk -> 2, confidence -> 0.7705}, []}, {chunk, 478, 508, single nucleotide polymorphisms, {entity -> TREATMENT, sentence -> 2, chunk -> 3, confidence -> 0.7204666}, []}, {chunk, 608, 625, insertion/deletion, {entity -> PROBLEM, sentence -> 3, chunk -> 6, confidence -> 0.9841}, []}]|

```
As you can see, only the treatment entities with confidence score of more than 0.7, and the problem entities with confidence score of more than 0.9 have been kept in the output.


#### Extended regex dictionary fuctionallity in Deidentification

The RegexPatternsDictionary can now use a regex that spawns the 2 previous token and the 2 next tokens.
That feature is implemented using regex groups.

Examples:

Given the sentence `The patient with ssn 123123123` we can use the following regex to capture the entitty `ssn (\d{9})`
Given the sentence `The patient has 12 years` we can use the following regex to capture the entitty `(\d{2}) years`

#### Enhanced RelationExtractionDL Model to create and identify relations between entities across the entire document

A new option has been added to `RENerChunksFilter` to support pairing entities from different sentences using `.setDocLevelRelations(True)`, to pass to the Relation Extraction Model. The RelationExtractionDL Model has also been updated to process document-level relations.

How to use:

```python
        re_dl_chunks = RENerChunksFilter() \
            .setInputCols(["ner_chunks", "dependencies"])\
            .setDocLevelRelations(True)\
            .setMaxSyntacticDistance(7)\
            .setOutputCol("redl_ner_chunks")  
```

Examples:

Given a document containing multiple sentences: `John somkes cigrettes. He also consumes alcohol.`, now we can generate relation pairs across sentences and relate `alcohol` with `John` .

#### Set NER graph explicitely in MedicalNerApproach
Now MedicalNerApproach can receives the path to the graph directly. When a graph location is provided through this method, previous graph search behavior is disabled.
```
MedicalNerApproach.setGraphFile(graphFilePath)
```

#### MedicalNerApproach can now accept a user-defined name for log file.
Now MedicalNerApproach can accept a user-defined name for the log file. If not such a name is provided, the conventional naming will take place.
```
MedicalNerApproach.setLogPrefix("oncology_ner")
```
This will result in `oncology_ner_20210605_141701.log` filename being used, in which the `20210605_141701` is a timestamp.

#### New Notebooks

+ A [new notebook](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.4.Biomedical_NER_SparkNLP_paper_reproduce.ipynb)  to reproduce our peer-reviewed NER paper (https://arxiv.org/abs/2011.06315)
+ New databricks [case study notebooks](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/databricks/python/healthcare_case_studies). In these notebooks, we showed the examples of how to work with oncology notes dataset and OCR on databricks for both DBr and community edition versions.


#### Updated Resolver Models
We updated `sbiobertresolve_snomed_findings` and `sbiobertresolve_cpt_procedures_augmented` resolver models to reflect the latest changes in the official terminologies.

#### Getting Started with Spark NLP for Healthcare Notebook in Databricks
We prepared a new notebook for those who want to get started with Spark NLP for Healthcare in Databricks : [Getting Started with Spark NLP for Healthcare Notebook](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/databricks/python/healthcare_case_studies/Get_Started_Spark_NLP_for_Healthcare.ipynb)

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_0_3">Version 3.0.3</a>
    </li>
    <li>
        <strong>Version 3.1.0</strong>
    </li>
    <li>
        <a href="release_notes_3_1_3">Version 3.1.1</a>
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
    <li class="active"><a href="release_notes_3_1_0">3.1.0</a></li>
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