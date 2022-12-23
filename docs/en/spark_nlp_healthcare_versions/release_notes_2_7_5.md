---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 2.7.5
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_2_7_5
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="prev_ver h3-box" markdown="1">

### 2.7.5

We are glad to announce that Spark NLP for Healthcare 2.7.5 has been released!  

#### Highlights:

- New pretrained **Relation Extraction** model to link clinical tests to test results and dates to clinical entities: `re_test_result_date`
- Adding two new `Admission` and `Discharge` entities to `ner_events_clinical` and renaming it to `ner_events_admission_clinical`
- Improving `ner_deid_enriched` NER model to cover `Doctor` and `Patient` name entities in various context and notations.
- Bug fixes & general improvements.

#### 1. re_test_result_date :

text = "Hospitalized with pneumonia in June, confirmed by a positive PCR of any specimen, evidenced by SPO2 </= 93% or PaO2/FiO2 < 300 mmHg"

|    | Chunk-1   | Entity-1   | Chunk-2   | Entity-2    | Relation     |
|---:|:----------|:-----------|:----------|:------------|:-------------|
|  0 | pneumonia | Problem    | june      | Date        | is_date_of   |
|  1 | PCR       | Test       | positive  | Test_Result | is_result_of |
|  2 | SPO2      | Test       | 93%       | Test_Result | is_result_of |
|  3 | PaO2/FiO2 | Test       | 300 mmHg  | Test_Result | is_result_of |

#### 2. `ner_events_admission_clinical` :

`ner_events_clinical` NER model is updated & improved to include `Admission` and `Discharge` entities.

text ="She is diagnosed as cancer in 1991. Then she was admitted to Mayo Clinic in May 2000 and discharged in October 2001"

|    | chunk        | entity        |
|---:|:-------------|:--------------|
|  0 | diagnosed    | OCCURRENCE    |
|  1 | cancer       | PROBLEM       |
|  2 | 1991         | DATE          |
|  3 | admitted     | ADMISSION     |
|  4 | Mayo Clinic  | CLINICAL_DEPT |
|  5 | May 2000     | DATE          |
|  6 | discharged   | DISCHARGE     |
|  7 | October 2001 | DATE          |


#### 3. Improved `ner_deid_enriched` :

PHI NER model is retrained to cover `Doctor` and `Patient` name entities even there is a punctuation between tokens as well as all upper case or lowercased.

text ="A . Record date : 2093-01-13 , DAVID HALE , M.D . , Name : Hendrickson , Ora MR . # 7194334 Date : 01/13/93 PCP : Oliveira , 25 month years-old , Record date : 2079-11-09 . Cocke County Baptist Hospital . 0295 Keats Street"

|    | chunk                         | entity        |
|---:|:------------------------------|:--------------|
|  0 | 2093-01-13                    | MEDICALRECORD |
|  1 | DAVID HALE                    | DOCTOR        |
|  2 | Hendrickson , Ora             | PATIENT       |
|  3 | 7194334                       | MEDICALRECORD |
|  4 | 01/13/93                      | DATE          |
|  5 | Oliveira                      | DOCTOR        |
|  6 | 25                            | AGE           |
|  7 | 2079-11-09                    | MEDICALRECORD |
|  8 | Cocke County Baptist Hospital | HOSPITAL      |
|  9 | 0295 Keats Street             | STREET        |


</div><div class="prev_ver h3-box" markdown="1">

## Versions

</div>
{%- include docs-healthcare-pagination.html -%}