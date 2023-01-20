---
layout: docs
header: true
seotitle: Spark NLP | John Snow Labs
title: Risk Adjustments Score Calculation
permalink: /docs/en/healthcare_risk_adjustments_score_calculation
key: docs-licensed-risk-adjustments-score-calculation
modify_date: "2022-12-26"
use_language_switcher: "Python-Scala"
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

<div class="h3-box" markdown="1">

Our Risk Adjustment Score implementation uses the Hierarchical Condition Category (HCC) Risk Adjustment model from the Centers for Medicare & Medicaid Service (CMS). HCC groups similar conditions in terms of healthcare costs and similarities in the diagnosis, and the model uses any ICD code that has a corresponging HCC category in the computation, discarding other ICD codes.

This module supports versions 22, 23, and 24 of the CMS-HCC risk adjustment model and needs the following parameters in order to calculate the risk score:

- ICD Codes (Obtained by, e.g., our pretrained model `sbiobertresolve_icd10cm_augmented_billable_hcc` from the[SentenceEntityResolverModel](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators#sentenceentityresolver) annotator)
- Age (Obtained by, e.g., our pretrained model `ner_jsl` from the[NerModel](https://nlp.johnsnowlabs.com/docs/en/licensed_annotators#nermodel) annotator)
- Gender (Obtained by, e.g., our pretrained model `classifierdl_gender_biobert` from the[ClassifierDLModel](https://nlp.johnsnowlabs.com/docs/en/annotators#classifierdl) annotator)
- The eligibility segment of the patient (information from the health plan provider)
- The original reason for entitlement (information from the health plan provider)
- If the patient is in Medicaid or not (information from the health plan provider)

## Available softwares and profiles

As mentioned, we implemented versions 22, 23, and 24 of the CMS-HCC software, and have the following profiles:

- Version 22
  - Year 2017
  - Year 2018
  - Year 2019
  - Year 2020
  - Year 2021
  - Year 2022
- Version 23
  - Year 2018
  - Year 2019
- Version 24
  - Year 2017
  - Year 2018
  - Year 2019
  - Year 2020
  - Year 2021
  - Year 2022
                   
## Usage

The module can perform the computations given a data frame containing the required information (Age, Gender, ICD codes, eligibility segment, the original reason for entitlement, and if the patient is in Medicaid or not). For example, given the dataset `df`:

```
+--------------+---+--------------------+-------------------------------+------+-----------+----+--------+----------+
|      filename|Age|          icd10_code|Extracted_Entities_vs_ICD_Codes|Gender|eligibility|orec|medicaid|       DOB|
+--------------+---+--------------------+-------------------------------+------+-----------+----+--------+----------+
|mt_note_03.txt| 66|[C499, C499, D618...|           [{leiomyosarcoma,...|     F|        CND|   1|   false|1956-05-30|
|mt_note_01.txt| 59|              [C801]|               [{cancer, C801}]|     F|        CFA|   0|    true|1961-10-12|
|mt_note_10.txt| 16|      [C6960, C6960]|           [{Rhabdomyosarcom...|     M|        CFA|   2|   false|2006-02-14|
|mt_note_08.txt| 66|        [C459, C800]|           [{malignant mesot...|     F|        CND|   1|    true|1956-03-17|
|mt_note_09.txt| 19|      [D5702, K5505]|           [{Sickle cell cri...|     M|        CPA|   3|    true|2003-06-11|
|mt_note_05.txt| 57|[C5092, C5091, C5...|           [{Breast Cancer, ...|     F|        CPA|   3|    true|1963-08-12|
|mt_note_06.txt| 63|        [F319, F319]|           [{type 1 bipolar ...|     F|        CFA|   0|   false|1959-07-24|
+--------------+---+--------------------+-------------------------------+------+-----------+----+--------+----------+
```
Where column `orec` means original reason for entitlement and `DOB` means date of birth (can also be used to compute age). You can use any of the available profiles to compute the scores (in the example, we use version 24, year 2020):

```python
from johnsnowlabs import medical

# Creates the risk profile
df = df.withColumn(
    "hcc_profile",
    medical.profileV24Y20(
        df.icd10_code, 
        df.Age, 
        df.Gender, 
        df.eligibility, 
        df.orec, 
        df.medicaid
    ),
)

# Extract relevant information
df = (
    df.withColumn("risk_score", df.hcc_profile.getItem("risk_score"))
    .withColumn("hcc_lst", df.hcc_profile.getItem("hcc_lst"))
    .withColumn("parameters", df.hcc_profile.getItem("parameters"))
    .withColumn("details", df.hcc_profile.getItem("details"))
)
df.select(
    "filename",
    "risk_score",
    "icd10_code",
    "Age",
    "Gender",
    "eligibility",
    "orec",
    "medicaid",
).show(truncate=False)
```

```
+--------------+----------+---------------------------------------------+---+------+-----------+----+--------+
filename      |risk_score|icd10_code                                   |Age|Gender|eligibility|orec|medicaid|
+--------------+----------+---------------------------------------------+---+------+-----------+----+--------+
mt_note_01.txt|0.158     |[C801]                                       |59 |F     |CFA        |0   |true    |
mt_note_03.txt|1.03      |[C499, C499, D6181, M069, C801]              |66 |F     |CND        |1   |false   |
mt_note_05.txt|2.991     |[C5092, C5091, C779, C5092, C800, G20, C5092]|57 |F     |CPA        |3   |true    |
mt_note_06.txt|0.299     |[F319]                                       |63 |F     |CFA        |0   |true    |
mt_note_08.txt|2.714     |[C459, C800]                                 |66 |F     |CND        |1   |false   |
mt_note_09.txt|1.234     |[D5702, K5505]                               |19 |F     |CPA        |3   |true    |
+--------------+----------+---------------------------------------------+---+------+-----------+----+--------+
```


For more details and usage examples, check the notebook [Medicare Risk Adjustment](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings_JSL/Healthcare/3.1.Calculate_Medicare_Risk_Adjustment_Score.ipynb) notebook from our Spark NLP Workshop repository.


</div>
