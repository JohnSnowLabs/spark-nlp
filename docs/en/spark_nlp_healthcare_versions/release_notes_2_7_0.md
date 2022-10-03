---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.7.0
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_7_0
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

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


<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_2_6_2">Version 2.6.2</a>
    </li>
    <li>
        <strong>Version 2.7.0</strong>
    </li>
    <li>
        <a href="release_notes_2_7_1">Version 2.7.1</a>
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
    <li><a href="release_notes_2_7_3">2.7.3</a></li>
    <li><a href="release_notes_2_7_2">2.7.2</a></li>
    <li><a href="release_notes_2_7_1">2.7.1</a></li>
    <li class="active"><a href="release_notes_2_7_0">2.7.0</a></li>
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