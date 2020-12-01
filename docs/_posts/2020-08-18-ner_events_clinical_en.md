---
layout: model
title: Detect Clinical Events
author: John Snow Labs
name: ner_events_clinical
class: NerDLModel
language: en
repository: clinical/models
date: 2020-08-18
tags: [clinical,licensed,ner,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---
 
{:.h2_title}
## Description
Pretrained named entity recognition deep learning model for clinical events.

## Predicted Entities 
`CLINICAL_DEPT`, `DATE`, `DURATION`, `EVIDENTIAL`, `FREQUENCY`, `OCCURRENCE`, `PROBLEM`, `TEST`, `TIME`, `TREATMENT`

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
nlpPipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings_clinical, licensed,model, ner_converter])

model = nlpPipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

results = model.transform(spark.createDataFrame(pd.DataFrame({"text": ["""This is the case of a very pleasant 46-year-old Caucasian female with subarachnoid hemorrhage secondary to ruptured left posteroinferior cerebellar artery aneurysm, which was clipped. The patient last underwent a right frontal ventricular peritoneal shunt on 10/12/07. This resulted in relief of left chest pain, but the patient continued to complaint of persistent pain to the left shoulder and left elbow. She was seen in clinic on 12/11/07 during which time MRI of the left shoulder showed no evidence of rotator cuff tear. She did have a previous MRI of the cervical spine that did show an osteophyte on the left C6-C7 level. Based on this, negative MRI of the shoulder, the patient was recommended to have anterior cervical discectomy with anterior interbody fusion at C6-C7 level. Operation, expected outcome, risks, and benefits were discussed with her. Risks include, but not exclusive of bleeding and infection, bleeding could be soft tissue bleeding, which may compromise airway and may result in return to the operating room emergently for evacuation of said hematoma. There is also the possibility of bleeding into the epidural space, which can compress the spinal cord and result in weakness and numbness of all four extremities as well as impairment of bowel and bladder function. Should this occur, the patient understands that she needs to be brought emergently back to the operating room for evacuation of said hematoma. There is also the risk of infection, which can be superficial and can be managed with p.o. antibiotics. However, the patient may develop deeper-seated infection, which may require return to the operating room. Should the infection be in the area of the spinal instrumentation, this will cause a dilemma since there might be a need to remove the spinal instrumentation and/or allograft. There is also the possibility of potential injury to the esophageus, the trachea, and the carotid artery. There is also the risks of stroke on the right cerebral circulation should an undiagnosed plaque be propelled from the right carotid. There is also the possibility hoarseness of the voice secondary to injury to the recurrent laryngeal nerve. There is also the risk of pseudoarthrosis and hardware failure. She understood all of these risks and agreed to have the procedure performed."""]})))
```

```scala
...

val model = NerDLModel.pretrained("ner_events_clinical","en","clinical/models")
    .setInputCols("sentence","token","word_embeddings")
    .setOutputCol("ner")

...

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings_clinical, licensed,model, ner_converter))

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
Trained using ``clinical_embeddings`` on i2b2 events data available from https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/

## Benchmarking
```bash
|    | label           |     tp |    fp |   fn |     prec |      rec |        f1 |
|---:|----------------:|-------:|------:|-----:|---------:|---------:|----------:|
|  0 | I-TIME          |     82 |    12 |   45 | 0.87234  | 0.645669 | 0.742081  |
|  1 | I-TREATMENT     |   2580 |   439 |  535 | 0.854588 | 0.82825  | 0.841213  |
|  2 | B-OCCURRENCE    |   1548 |   680 |  945 | 0.694793 | 0.620939 | 0.655793  |
|  3 | I-DURATION      |    366 |   183 |   99 | 0.666667 | 0.787097 | 0.721893  |
|  4 | B-DATE          |    847 |   151 |  138 | 0.848697 | 0.859898 | 0.854261  |
|  5 | I-DATE          |    921 |   191 |  196 | 0.828237 | 0.82453  | 0.82638   |
|  6 | B-ADMISSION     |    105 |   102 |   15 | 0.507246 | 0.875    | 0.642202  |
|  7 | I-PROBLEM       |   5238 |   902 |  823 | 0.853094 | 0.864214 | 0.858618  |
|  8 | B-CLINICAL_DEPT |    613 |   130 |  119 | 0.825034 | 0.837432 | 0.831187  |
|  9 | B-TIME          |     36 |     8 |   24 | 0.818182 | 0.6      | 0.692308  |
| 10 | I-CLINICAL_DEPT |   1273 |   210 |  137 | 0.858395 | 0.902837 | 0.880055  |
| 11 | B-PROBLEM       |   3717 |   608 |  591 | 0.859422 | 0.862813 | 0.861114  |
| 12 | I-TEST          |   2304 |   384 |  361 | 0.857143 | 0.86454  | 0.860826  |
| 13 | B-TEST          |   1870 |   372 |  300 | 0.834077 | 0.861751 | 0.847688  |
| 14 | B-TREATMENT     |   2767 |   437 |  513 | 0.863608 | 0.843598 | 0.853485  |
| 15 | B-EVIDENTIAL    |    394 |   109 |  201 | 0.7833   | 0.662185 | 0.717669  |
| 16 | B-DURATION      |    236 |   119 |  105 | 0.664789 | 0.692082 | 0.678161  |
| 17 | B-FREQUENCY     |    117 |    20 |   79 | 0.854015 | 0.596939 | 0.702703  |
| 18 | Macro-average   | 25806  | 5821  | 6342 | 0.735285 | 0.677034 | 0.704959  |
| 19 | Micro-average   | 25806  | 5821  | 6342 | 0.815948 | 0.802725 | 0.809283  |
```