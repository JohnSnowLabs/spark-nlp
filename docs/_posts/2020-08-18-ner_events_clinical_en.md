---
layout: model
title: Ner DL Model Events
author: John Snow Labs
name: ner_events_clinical
class: NerDLModel
language: en
repository: clinical/models
date: 2020-08-18
tags: [clinical,ner,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---
 
{:.h2_title}
## Description
Pretrained named entity recognition deep learning model for clinical events.

## Predicted Entities 
Clinical_Dept, Date, Duration, Evidential, Frequency, Occurrence, Problem, Test, Time, Treatment

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_EVENTS_CLINICAL/){:.button.button-orange.button-orange-trans.co.button-icon}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_EVENTS_CLINICAL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_events_clinical_en_2.5.5_2.4_1597775531760.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...

model = NerDLModel.pretrained("ner_events_clinical","en","clinical/models")\
    .setInputCols("sentence","token","word_embeddings")\
    .setOutputCol("ner")

...

nlpPipeline = Pipeline(stages=[document_assembler,
                               sentence_detector,
                               tokenizer,
                               embeddings_clinical,
                               model,
                               ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""This is the case of a very pleasant 46-year-old Caucasian female with subarachnoid hemorrhage secondary to ruptured left posteroinferior cerebellar artery aneurysm, which was clipped. The patient last underwent a right frontal ventricular peritoneal shunt on 10/12/07. This resulted in relief of left chest pain, but the patient continued to complaint of persistent pain to the left shoulder and left elbow. She was seen in clinic on 12/11/07 during which time MRI of the left shoulder showed no evidence of rotator cuff tear. She did have a previous MRI of the cervical spine that did show an osteophyte on the left C6-C7 level. Based on this, negative MRI of the shoulder, the patient was recommended to have anterior cervical discectomy with anterior interbody fusion at C6-C7 level. Operation, expected outcome, risks, and benefits were discussed with her. Risks include, but not exclusive of bleeding and infection, bleeding could be soft tissue bleeding, which may compromise airway and may result in return to the operating room emergently for evacuation of said hematoma. There is also the possibility of bleeding into the epidural space, which can compress the spinal cord and result in weakness and numbness of all four extremities as well as impairment of bowel and bladder function. Should this occur, the patient understands that she needs to be brought emergently back to the operating room for evacuation of said hematoma. There is also the risk of infection, which can be superficial and can be managed with p.o. antibiotics. However, the patient may develop deeper-seated infection, which may require return to the operating room. Should the infection be in the area of the spinal instrumentation, this will cause a dilemma since there might be a need to remove the spinal instrumentation and/or allograft. There is also the possibility of potential injury to the esophageus, the trachea, and the carotid artery. There is also the risks of stroke on the right cerebral circulation should an undiagnosed plaque be propelled from the right carotid. There is also the possibility hoarseness of the voice secondary to injury to the recurrent laryngeal nerve. There is also the risk of pseudoarthrosis and hardware failure. She understood all of these risks and agreed to have the procedure performed."""]})))
```

```scala
...

val model = NerDLModel.pretrained("ner_events_clinical","en","clinical/models")
    .setInputCols("sentence","token","word_embeddings")
    .setOutputCol("ner")

...

val pipeline = new Pipeline().setStages(Array(
                               document_assembler,
                               sentence_detector,
                               tokenizer,
                               embeddings_clinical,
                               model,
                               ner_converter))

val result = pipeline.fit(Seq.empty["""This is the case of a very pleasant 46-year-old Caucasian female with subarachnoid hemorrhage secondary to ruptured left posteroinferior cerebellar artery aneurysm, which was clipped. The patient last underwent a right frontal ventricular peritoneal shunt on 10/12/07. This resulted in relief of left chest pain, but the patient continued to complaint of persistent pain to the left shoulder and left elbow. She was seen in clinic on 12/11/07 during which time MRI of the left shoulder showed no evidence of rotator cuff tear. She did have a previous MRI of the cervical spine that did show an osteophyte on the left C6-C7 level. Based on this, negative MRI of the shoulder, the patient was recommended to have anterior cervical discectomy with anterior interbody fusion at C6-C7 level. Operation, expected outcome, risks, and benefits were discussed with her. Risks include, but not exclusive of bleeding and infection, bleeding could be soft tissue bleeding, which may compromise airway and may result in return to the operating room emergently for evacuation of said hematoma. There is also the possibility of bleeding into the epidural space, which can compress the spinal cord and result in weakness and numbness of all four extremities as well as impairment of bowel and bladder function. Should this occur, the patient understands that she needs to be brought emergently back to the operating room for evacuation of said hematoma. There is also the risk of infection, which can be superficial and can be managed with p.o. antibiotics. However, the patient may develop deeper-seated infection, which may require return to the operating room. Should the infection be in the area of the spinal instrumentation, this will cause a dilemma since there might be a need to remove the spinal instrumentation and/or allograft. There is also the possibility of potential injury to the esophageus, the trachea, and the carotid artery. There is also the risks of stroke on the right cerebral circulation should an undiagnosed plaque be propelled from the right carotid. There is also the possibility hoarseness of the voice secondary to injury to the recurrent laryngeal nerve. There is also the risk of pseudoarthrosis and hardware failure. She understood all of these risks and agreed to have the procedure performed."""].toDS.toDF("text")).transform(data)

```
</div>

{:.h2_title}
## Results
The output is a dataframe with a sentence per row and a `"ner"` column containing all of the entity labels in the sentence, entity character indices, and other metadata. To get only the tokens and entity labels, without the metadata, select `"token.result"` and `"ner.result"` from your output dataframe or add the `"Finisher"` to the end of your pipeline.

```bash
+--------------------------------------------------------+---------+
|chunk                                                   |ner      |
+--------------------------------------------------------+---------+
|subarachnoid hemorrhage                                 |PROBLEM  |
|ruptured left posteroinferior cerebellar artery aneurysm|PROBLEM  |
|clipped                                                 |TREATMENT|
|a right frontal ventricular peritoneal shunt            |TREATMENT|
|left chest pain                                         |PROBLEM  |
|persistent pain to the left shoulder and left elbow     |PROBLEM  |
|MRI of the left shoulder                                |TEST     |
|rotator cuff tear                                       |PROBLEM  |
|a previous MRI of the cervical spine                    |TEST     |
|an osteophyte on the left C6-C7 level                   |PROBLEM  |
|MRI of the shoulder                                     |TEST     |
|anterior cervical discectomy                            |TREATMENT|
|anterior interbody fusion                               |TREATMENT|
|bleeding                                                |PROBLEM  |
|infection                                               |PROBLEM  |
|bleeding                                                |PROBLEM  |
|soft tissue bleeding                                    |PROBLEM  |
|evacuation                                              |TREATMENT|
|hematoma                                                |PROBLEM  |
|bleeding into the epidural space                        |PROBLEM  |
+--------------------------------------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|----------------|----------------------------------|
| Name:           | ner_events_clinical              |
| Type:    | NerDLModel                       |
| Compatibility:  | Spark NLP 2.5.0+                            |
| License:        | Licensed                         |
|Edition:|Official|                       |
|Input labels:         | [sentence, token, word_embeddings] |
|Output labels:        | [ner]                              |
| Language:       | en                               |
| Case sensitive: | False                            |
| Dependencies:  | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on i2b2 events data with `clinical_embeddings` at https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

## Benchmarking
```bash
|    | label         |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|--------------:|------:|------:|------:|---------:|---------:|---------:|
|  0 | I-NAME        |  1096 |    47 |    80 | 0.95888  | 0.931973 | 0.945235 |
|  1 | I-CONTACT     |    93 |     0 |     4 | 1        | 0.958763 | 0.978947 |
|  2 | I-AGE         |     3 |     1 |     6 | 0.75     | 0.333333 | 0.461538 |
|  3 | B-DATE        |  2078 |    42 |    52 | 0.980189 | 0.975587 | 0.977882 |
|  4 | I-DATE        |   474 |    39 |    25 | 0.923977 | 0.9499   | 0.936759 |
|  5 | I-LOCATION    |   755 |    68 |    76 | 0.917375 | 0.908544 | 0.912938 |
|  6 | I-PROFESSION  |    78 |     8 |     9 | 0.906977 | 0.896552 | 0.901734 |
|  7 | B-NAME        |  1182 |   101 |    36 | 0.921278 | 0.970443 | 0.945222 |
|  8 | B-AGE         |   259 |    10 |    11 | 0.962825 | 0.959259 | 0.961039 |
|  9 | B-ID          |   146 |     8 |    11 | 0.948052 | 0.929936 | 0.938907 |
| 10 | B-PROFESSION  |    76 |     9 |    21 | 0.894118 | 0.783505 | 0.835165 |
| 11 | B-LOCATION    |   556 |    87 |    71 | 0.864697 | 0.886762 | 0.875591 |
| 12 | I-ID          |    64 |     8 |     3 | 0.888889 | 0.955224 | 0.920863 |
| 13 | B-CONTACT     |    40 |     7 |     5 | 0.851064 | 0.888889 | 0.869565 |
| 14 | Macro-average |  6900 |   435 |   410 | 0.912023 | 0.880619 | 0.896046 |
| 15 | Micro-average |  6900 |   435 |   410 | 0.940695 | 0.943912 | 0.942301 |
```