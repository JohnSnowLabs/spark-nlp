---
layout: model
title: Detect Clinical Relations 
author: John Snow Labs
name: re_clinical
class: RelationExtractionModel
language: en
repository: clinical/models
date: 2020-09-24
task: Relation Extraction
edition: Healthcare NLP 2.5.5
spark_version: 2.4
tags: [clinical,licensed,relation extraction,en]
supported: true
annotator: RelationExtractionModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Relation Extraction model based on syntactic features using deep learning. Models the set of clinical relations defined in the 2010 ``i2b2`` relation challenge. 
`TrIP`: A certain treatment has improved or cured a medical problem (eg, ‘infection resolved with antibiotic course’)
`TrWP`: A patient's medical problem has deteriorated or worsened because of or in spite of a treatment being administered (eg, ‘the tumor was growing despite the drain’)
`TrCP`: A treatment caused a medical problem (eg, ‘penicillin causes a rash’)
`TrAP`: A treatment administered for a medical problem (eg, ‘Dexamphetamine for narcolepsy’)
`TrNAP`: The administration of a treatment was avoided because of a medical problem (eg, ‘Ralafen which is contra-indicated because of ulcers’)
`TeRP`: A test has revealed some medical problem (eg, ‘an echocardiogram revealed a pericardial effusion’)
`TeCP`: A test was performed to investigate a medical problem (eg, ‘chest x-ray done to rule out pneumonia’)
`PIP`: Two problems are related to each other (eg, ‘Azotemia presumed secondary to sepsis’)

{:.h2_title}
## Predicted Entities
 
`TrIP`, `TrWP`, `TrCP`, `TrAP`, `TrAP`, `TeRP`, `TeCP`, `PIP`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/RE_CLINICAL/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/10.Clinical_Relation_Extraction.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/re_clinical_en_2.5.5_2.4_1600987935304.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/re_clinical_en_2.5.5_2.4_1600987935304.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

{:.h2_title}
## How to use 

In the table below, `re_clinical` RE model, its labels, optimal NER model, and meaningful relation pairs are illustrated.

 |   RE MODEL  | RE MODEL LABES                        |   NER MODEL  | RE PAIRS                  |
 |:-----------:|----------------------------------------|:------------:|---------------------------|
 | re_clinical | TrIP,TrWP,TrCP,TrAP,TrAP,TeRP,TeCP,PIP | ner_clinical | [“No need to set pairs.”] |

 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...

reModel = RelationExtractionModel.pretrained("re_clinical","en","clinical/models")\
    .setInputCols(["word_embeddings","chunk","pos","dependency"])\
    .setOutputCol("relations")

pipeline = Pipeline(stages=[documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, reModel])
model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = LightPipeline(model).fullAnnotate("""A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation,  associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation. Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge .""")
```

```scala
...

val reModel = RelationExtractionModel.pretrained("re_clinical","en","clinical/models")
    .setInputCols("word_embeddings","chunk","pos","dependency")
    .setOutputCol("relations")
    
val pipeline = new Pipeline().setStages(Array(documenter, sentencer, tokenizer, words_embedder, pos_tagger, ner_tagger, ner_chunker, dependency_parser, reModel))

val data = Seq("A 28-year-old female with a history of gestational diabetes mellitus diagnosed eight years prior to presentation and subsequent type two diabetes mellitus ( T2DM ), one prior episode of HTG-induced pancreatitis three years prior to presentation,  associated with an acute hepatitis , and obesity with a body mass index ( BMI ) of 33.5 kg/m2 , presented with a one-week history of polyuria , polydipsia , poor appetite , and vomiting . Two weeks prior to presentation , she was treated with a five-day course of amoxicillin for a respiratory tract infection . She was on metformin , glipizide , and dapagliflozin for T2DM and atorvastatin and gemfibrozil for HTG . She had been on dapagliflozin for six months at the time of presentation. Physical examination on presentation was significant for dry oral mucosa ; significantly , her abdominal examination was benign with no tenderness , guarding , or rigidity . Pertinent laboratory findings on admission were : serum glucose 111 mg/dl , bicarbonate 18 mmol/l , anion gap 20 , creatinine 0.4 mg/dL , triglycerides 508 mg/dL , total cholesterol 122 mg/dL , glycated hemoglobin ( HbA1c ) 10% , and venous pH 7.27 . Serum lipase was normal at 43 U/L . Serum acetone levels could not be assessed as blood samples kept hemolyzing due to significant lipemia . The patient was initially admitted for starvation ketosis , as she reported poor oral intake for three days prior to admission . However , serum chemistry obtained six hours after presentation revealed her glucose was 186 mg/dL , the anion gap was still elevated at 21 , serum bicarbonate was 16 mmol/L , triglyceride level peaked at 2050 mg/dL , and lipase was 52 U/L . The β-hydroxybutyrate level was obtained and found to be elevated at 5.29 mmol/L - the original sample was centrifuged and the chylomicron layer removed prior to analysis due to interference from turbidity caused by lipemia again . The patient was treated with an insulin drip for euDKA and HTG with a reduction in the anion gap to 13 and triglycerides to 1400 mg/dL , within 24 hours . Her euDKA was thought to be precipitated by her respiratory tract infection in the setting of SGLT2 inhibitor use . The patient was seen by the endocrinology service and she was discharged on 40 units of insulin glargine at night , 12 units of insulin lispro with meals , and metformin 1000 mg two times a day . It was determined that all SGLT2 inhibitors should be discontinued indefinitely . She had close follow-up with endocrinology post discharge .").toDF("text")
val result = pipeline.fit(data).transform(data)

```
</div>

{:.h2_title}
## Results
```bash
| relation | entity1   | chunk1                          | entity2    | chunk2                        | confidence |
|----------|-----------|---------------------------------|------------|-------------------------------|------------|
| TrAP     | PROBLEM   | T2DM                            | TREATMENT  | atorvastatin                  | 0.99955326 |
| TrWP     | TEST      | blood samples                   | PROBLEM    | significant lipemia           | 0.99998724 |
| TeRP     | TEST      | the anion gap                   | PROBLEM    | still elevated                | 0.9965193  |
| TrAP     | TEST      | analysis                        | PROBLEM    | interference from turbidity   | 0.9676019  |
| TrWP     | TREATMENT | an insulin drip                 | PROBLEM    | a reduction in the anion gap  | 0.94099987 |
| TeRP     | PROBLEM   | a reduction in the anion gap    | TEST       | triglycerides                 | 0.9956793  |
| TeRP     | PROBLEM   | her respiratory tract infection | TREATMENT  | SGLT2 inhibitor               | 0.997498   |
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|-----------------------------------------|
| Name:           | re_clinical                             |
| Type:    | RelationExtractionModel                 |
| Compatibility:  | Spark NLP 2.5.5+                                   |
| License:        | Licensed                                |
|Edition:|Official|                              |
|Input labels:         | [word_embeddings, chunk, pos, dependency] |
|Output labels:        | [category]                                |
| Language:       | en                                      |
| Case sensitive: | False                                   |
| Dependencies:  | embeddings_clinical                     |

{:.h2_title}
## Data Source
Trained on data gathered and manually annotated by John Snow Labs
https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

## Benchmarking
```bash
       label  precision  recall  f1-score
           O       0.96    0.93      0.94
        TeRP       0.91    0.94      0.92
         PIP       0.86    0.92      0.89
        TrAP       0.81    0.92      0.86
        TrCP       0.56    0.55      0.55
        TeCP       0.57    0.49      0.53
    accuracy        -       -        0.88
   macro-avg       0.65    0.59      0.60
weighted-avg       0.87    0.88      0.87 
```