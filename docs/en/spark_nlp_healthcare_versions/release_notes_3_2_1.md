---
layout: docs
header: true
seotitle: Spark NLP for Healthcare | John Snow Labs
title: Spark NLP for Healthcare Release Notes 3.2.1
permalink: /docs/en/spark_nlp_healthcare_versions/release_notes_3_2_1
key: docs-licensed-release-notes
modify_date: 2021-07-14
show_nav: true
sidebar:
    nav: sparknlp-healthcare
---

## 3.2.1
We are glad to announce that Spark NLP Healthcare 3.2.1 has been released!.

#### Highlights

+ Deprecated ChunkEntityResolver.
+ New BERT-Based NER Models
+ HCC module added support for versions v22 and v23.
+ Updated Notebooks for resolvers and graph builders.
+ New TF Graph Builder.

#### New BERT-Based NER Models

We have two new BERT-based token classifier NER models. These models are the first clinical NER models that use the BertForTokenCLassification approach that was introduced in Spark NLP 3.2.0.

+ `bert_token_classifier_ner_clinical`: This model is BERT-based version of `ner_clinical` model. This new model is 4% better than the legacy NER model (MedicalNerModel) that is based on BiLSTM-CNN-Char architecture.

*Metrics*:

```
              precision    recall  f1-score   support

     PROBLEM       0.88      0.92      0.90     30276
        TEST       0.91      0.86      0.88     17237
   TREATMENT       0.87      0.88      0.88     17298
           O       0.97      0.97      0.97    202438

    accuracy                           0.95    267249
   macro avg       0.91      0.91      0.91    267249
weighted avg       0.95      0.95      0.95    267249

```

*Example*:

```bash
documentAssembler = DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer()\
       .setInputCols("sentence")\
       .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_clinical", "en", "clinical/models")\
       .setInputCols("token", "sentence")\
       .setOutputCol("ner")\
       .setCaseSensitive(True)

ner_converter = NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
       documentAssembler,
       sentenceDetector,
       tokenizer,
       tokenClassifier,
       ner_converter
  ])

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

text = 'A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge .'

res = p_model.transform(spark.createDataFrame([[text]]).toDF("text")).collect()

res[0]['label']
```

+ `bert_token_classifier_ner_jsl`: This model is BERT-based version of `ner_jsl` model. This new model is better than the legacy NER model (MedicalNerModel) that is based on BiLSTM-CNN-Char architecture.

*Metrics*:

```
                                    precision    recall  f1-score   support

                    Admission_Discharge       0.84      0.97      0.90       415
                                    Age       0.96      0.96      0.96      2434
                                Alcohol       0.75      0.83      0.79       145
                               Allergen       0.33      0.16      0.22        25
                                    BMI       1.00      0.77      0.87        26
                           Birth_Entity       1.00      0.17      0.29        12
                         Blood_Pressure       0.86      0.88      0.87       597
                Cerebrovascular_Disease       0.74      0.77      0.75       266
                          Clinical_Dept       0.90      0.92      0.91      2385
                   Communicable_Disease       0.70      0.59      0.64        85
                                   Date       0.95      0.98      0.96      1438
                           Death_Entity       0.83      0.83      0.83        59
                               Diabetes       0.95      0.95      0.95       350
                                   Diet       0.60      0.49      0.54       229
                              Direction       0.88      0.90      0.89      6187
              Disease_Syndrome_Disorder       0.90      0.89      0.89     13236
                                 Dosage       0.57      0.49      0.53       263
                                   Drug       0.91      0.93      0.92     15926
                               Duration       0.82      0.85      0.83      1218
                           EKG_Findings       0.64      0.70      0.67       325
                             Employment       0.79      0.85      0.82       539
           External_body_part_or_region       0.84      0.84      0.84      4805
                  Family_History_Header       1.00      1.00      1.00       889
                          Fetus_NewBorn       0.57      0.56      0.56       341
                                   Form       0.53      0.43      0.48        81
                              Frequency       0.87      0.90      0.88      1718
                                 Gender       0.98      0.98      0.98      5666
                                    HDL       0.60      1.00      0.75         6
                          Heart_Disease       0.88      0.88      0.88      2295
                                 Height       0.89      0.96      0.92       134
                         Hyperlipidemia       1.00      0.95      0.97       194
                           Hypertension       0.95      0.98      0.97       566
                        ImagingFindings       0.66      0.64      0.65       601
                      Imaging_Technique       0.62      0.67      0.64       108
                    Injury_or_Poisoning       0.85      0.83      0.84      1680
            Internal_organ_or_component       0.90      0.91      0.90     21318
                         Kidney_Disease       0.89      0.89      0.89       446
                                    LDL       0.88      0.97      0.92        37
                        Labour_Delivery       0.82      0.71      0.76       306
                         Medical_Device       0.89      0.93      0.91     12852
                 Medical_History_Header       0.96      0.97      0.96      1013
                               Modifier       0.68      0.60      0.64      1398
                          O2_Saturation       0.84      0.82      0.83       199
                                Obesity       0.96      0.98      0.97       130
                            Oncological       0.88      0.96      0.92      1635
                             Overweight       0.80      0.80      0.80        10
                         Oxygen_Therapy       0.91      0.92      0.92       231
                              Pregnancy       0.81      0.83      0.82       439
                              Procedure       0.91      0.91      0.91     14410
                Psychological_Condition       0.81      0.81      0.81       354
                                  Pulse       0.85      0.95      0.89       389
                         Race_Ethnicity       1.00      1.00      1.00       163
                    Relationship_Status       0.93      0.91      0.92        57
                           RelativeDate       0.83      0.86      0.84      1562
                           RelativeTime       0.74      0.79      0.77       431
                            Respiration       0.99      0.95      0.97       221
                                  Route       0.68      0.69      0.69       597
                         Section_Header       0.97      0.98      0.98     28580
  Sexually_Active_or_Sexual_Orientation       1.00      0.64      0.78        14
                                Smoking       0.83      0.90      0.86       225
                  Social_History_Header       0.95      0.99      0.97       825
                               Strength       0.71      0.55      0.62       227
                              Substance       0.85      0.81      0.83       193
                     Substance_Quantity       0.00      0.00      0.00        28
                                Symptom       0.84      0.86      0.85     23092
                            Temperature       0.94      0.97      0.96       410
                                   Test       0.84      0.88      0.86      9050
                            Test_Result       0.84      0.84      0.84      2766
                                   Time       0.90      0.81      0.86       140
                      Total_Cholesterol       0.69      0.95      0.80        73
                              Treatment       0.73      0.72      0.73       506
                          Triglycerides       0.83      0.80      0.81        30
                             VS_Finding       0.76      0.77      0.76       588
                                Vaccine       0.70      0.84      0.76        92
                     Vital_Signs_Header       0.95      0.98      0.97      2223
                                 Weight       0.88      0.89      0.88       306
                                      O       0.97      0.96      0.97    253164

                               accuracy                           0.94    445974
                              macro avg       0.82      0.82      0.81    445974
                           weighted avg       0.94      0.94      0.94    445974
```

*Example*:

```
documentAssembler = DocumentAssembler()\
       .setInputCol("text")\
       .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer()\
       .setInputCols("sentence")\
       .setOutputCol("token")

tokenClassifier = BertForTokenClassification.pretrained("bert_token_classifier_ner_jsl", "en", "clinical/models")\
       .setInputCols("token", "sentence")\
       .setOutputCol("ner")\
       .setCaseSensitive(True)

ner_converter = NerConverter()\
        .setInputCols(["sentence","token","ner"])\
        .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
       documentAssembler,
       sentenceDetector,
       tokenizer,
       tokenClassifier,
       ner_converter
  ])

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

text = 'A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation , associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation . Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge .'

res = p_model.transform(spark.createDataFrame([[text]]).toDF("text")).collect()

res[0]['label']
```


#### HCC module added support for versions v22 and v23

Now we can use the version 22 and the version 23 for the new HCC module to calculate CMS-HCC Risk Adjustment score.

Added the following parameters `elig`, `orec` and `medicaid` on the profiles functions. These parameters may not be stored in clinical notes, and may require to be imported from other sources.

```
elig : The eligibility segment of the patient.
       Allowed values are as follows:
       - "CFA": Community Full Benefit Dual Aged
       - "CFD": Community Full Benefit Dual Disabled
       - "CNA": Community NonDual Aged
       - "CND": Community NonDual Disabled
       - "CPA": Community Partial Benefit Dual Aged
       - "CPD": Community Partial Benefit Dual Disabled
       - "INS": Long Term Institutional
       - "NE": New Enrollee
       - "SNPNE": SNP NE

orec: Original reason for entitlement code.
      - "0": Old age and survivor's insurance
      - "1": Disability insurance benefits
      - "2": End-stage renal disease
      - "3": Both DIB and ESRD

medicaid: If the patient is in Medicaid or not.

```

Required parameters should be stored in Spark dataframe.

```python

df.show(truncate=False)

+---------------+------------------------------+---+------+-----------+----+--------+
|hcc_profileV24 |icd10_code                    |age|gender|eligibility|orec|medicaid|
+---------------+------------------------------+---+------+-----------+----+--------+
|{"hcc_lst":[...|[E1169, I5030, I509, E852]    |64 |F     |CFA        |0   |true    |
|{"hcc_lst":[...|[G629, D469, D6181]           |77 |M     |CND        |1   |false   |
|{"hcc_lst":[...|[D473, D473, D473, M069, C969]|16 |F     |CPA        |3   |true    |
+---------------+------------------------------+---+------+-----------+----+--------+

The content of the hcc_profileV24 column is a JSON-parsable string, like in the following example,
{
    "hcc_lst": [
        "HCC18",
        "HCC85_gDiabetesMellit",
        "HCC85",
        "HCC23",
        "D3"
    ],
    "details": {
        "CNA_HCC18": 0.302,
        "CNA_HCC85": 0.331,
        "CNA_HCC23": 0.194,
        "CNA_D3": 0.0,
        "CNA_HCC85_gDiabetesMellit": 0.0
    },
    "hcc_map": {
        "E1169": [
            "HCC18"
        ],
        "I5030": [
            "HCC85"
        ],
        "I509": [
            "HCC85"
        ],
        "E852": [
            "HCC23"
        ]
    },
    "risk_score": 0.827,
    "parameters": {
        "elig": "CNA",
        "age": 56,
        "sex": "F",
        "origds": false,
        "disabled": false,
        "medicaid": false
    }
}


```
We can import different CMS-HCC model versions as seperate functions and use them in the same program.

```python

from sparknlp_jsl.functions import profile,profileV22,profileV23

df = df.withColumn("hcc_profileV24", profile(df.icd10_code,
                                          df.age,
                                          df.gender,
                                          df.eligibility,
                                          df.orec,
                                          df.medicaid
                                          ))

df.withColumn("hcc_profileV22", profileV22(df.codes, df.age, df.sex,df.elig,df.orec,df.medicaid))
df.withColumn("hcc_profileV23", profileV23(df.codes, df.age, df.sex,df.elig,df.orec,df.medicaid))

```


```python
df.show(truncate=False)

+----------+------------------------------+---+------+-----------+----+--------+
|risk_score|icd10_code                    |age|gender|eligibility|orec|medicaid|
+----------+------------------------------+---+------+-----------+----+--------+
|0.922     |[E1169, I5030, I509, E852]    |64 |F     |CFA        |0   |true    |
|3.566     |[G629, D469, D6181]           |77 |M     |CND        |1   |false   |
|1.181     |[D473, D473, D473, M069, C969]|16 |F     |CPA        |3   |true    |
+----------+------------------------------+---+------+-----------+----+--------+
```

#### Updated Notebooks for resolvers and graph builders

+ We have updated the resolver notebooks on spark-nlp-workshop repo with new `BertSentenceChunkEmbeddings` annotator. This annotator lets users aggregate sentence embeddings and ner chunk embeddings to get more specific and accurate resolution codes. It works by averaging context and chunk embeddings to get contextual information. Input to this annotator is the context (sentence) and ner chunks, while the output is embedding for each chunk that can be fed to the resolver model. The `setChunkWeight` parameter can be used to control the influence of surrounding context. Example below shows the comparison of old vs new approach.


|text|ner_chunk|entity|icd10_code|all_codes|resolutions|icd10_code_SCE|all_codes_SCE|resolutions_SCE|
|-|-|-|-|-|-|-|-|-|
|Two weeks prior to presentation, she was treated with a five-day course of amoxicillin for a respiratory tract infection.|a respiratory tract infection|PROBLEM|J988|[J988, J069, A499, J22, J209,...]|[respiratory tract infection, upper respiratory tract infection, bacterial respiratory infection, acute respiratory infection, bronchial infection,...]|Z870|[Z870, Z8709, J470, J988, A499,...|[history of acute lower respiratory tract infection (situation), history of acute lower respiratory tract infection, bronchiectasis with acute lower respiratory infection, rti - respiratory tract infection, bacterial respiratory infection,...|

Here are the updated resolver notebooks:

> - [3.Clinical_Entity_Resolvers.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/3.Clinical_Entity_Resolvers.ipynb)
> - [24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.Improved_Entity_Resolvers_in_SparkNLP_with_sBert.ipynb)

You can also check for more examples of this annotator: [24.1.Improved_Entity_Resolution_with_SentenceChunkEmbeddings.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/24.1.Improved_Entity_Resolution_with_SentenceChunkEmbeddings.ipynb)

+ We have updated TF Graph builder notebook to show how to create TF graphs with TF2.x.

> Here is the updated notebook: [17.Graph_builder_for_DL_models.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/17.Graph_builder_for_DL_models.ipynb)  

**To see more, please check: [Spark NLP Healthcare Workshop Repo](https://github.com/JohnSnowLabs/spark-nlp-workshop/tree/master/tutorials/Certification_Trainings/Healthcare)**


#### New TF Graph Builder

TF graph builder to create graphs and train DL models for licensed annotators (MedicalNer, Relation Extraction, Assertion and Generic Classifier) is made compatible with TF2.x.

To see how to create TF Graphs, you can check here: [17.Graph_builder_for_DL_models.ipynb](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/17.Graph_builder_for_DL_models.ipynb)  

<div class="prev_ver h3-box" markdown="1">

## Versions

</div>

<ul class="pagination">
    <li>
        <a href="release_notes_3_2_0">Version 3.2.0</a>
    </li>
    <li>
        <strong>Version 3.2.1</strong>
    </li>
    <li>
        <a href="release_notes_3_2_2">Version 3.2.2</a>
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
    <li class="active"><a href="release_notes_3_2_1">3.2.1</a></li>
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