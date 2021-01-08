---
layout: docs
header: true
title: Spark NLP for Healthcare Release Notes
permalink: /docs/en/licensed_release_notes
key: docs-licensed-release-notes
modify_date: "2020-11-26"
---

<div class="h3-box" markdown="1">

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
