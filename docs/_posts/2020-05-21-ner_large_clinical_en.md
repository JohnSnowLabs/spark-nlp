---
layout: model
title: Ner DL Model Clinical (Large)
author: John Snow Labs
name: ner_large_clinical
class: NerDLModel
language: en
repository: clinical/models
date: 2020-05-21
tags: [clinical,ner,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Named Entity recognition annotator allows for a generic model to be trained by utilizing a deep learning algorithm (Char CNNs - BiLSTM - CRF - word embeddings) inspired on a former state of the art model for NER: Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM,CNN.

## Predicted Entities 
PROBLEM, TEST, TREATMENT.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_EVENTS_CLINICAL/){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.Clinical_Named_Entity_Recognition_Model.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_large_clinical_en_2.5.0_2.4_1590021302624.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...

model = NerDLModel.pretrained("ner_large_clinical","en","clinical/models")\
	.setInputCols(["sentence","token","word_embeddings"])\
	.setOutputCol("ner")

...

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, word_embeddings, model, ner_converter])

light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

result = pipeline_model.transform(spark.createDataFrame(pd.DataFrame({"text": [
    """This is the case of a very pleasant 46-year-old Caucasian female with subarachnoid hemorrhage secondary to ruptured left posteroinferior cerebellar artery aneurysm, which was clipped. The patient last underwent a right frontal ventricular peritoneal shunt on 10/12/07. This resulted in relief of left chest pain, but the patient continued to complaint of persistent pain to the left shoulder and left elbow. She was seen in clinic on 12/11/07 during which time MRI of the left shoulder showed no evidence of rotator cuff tear. She did have a previous MRI of the cervical spine that did show an osteophyte on the left C6-C7 level. Based on this, negative MRI of the shoulder, the patient was recommended to have anterior cervical discectomy with anterior interbody fusion at C6-C7 level. Operation, expected outcome, risks, and benefits were discussed with her. Risks include, but not exclusive of bleeding and infection, bleeding could be soft tissue bleeding, which may compromise airway and may result in return to the operating room emergently for evacuation of said hematoma. There is also the possibility of bleeding into the epidural space, which can compress the spinal cord and result in weakness and numbness of all four extremities as well as impairment of bowel and bladder function. Should this occur, the patient understands that she needs to be brought emergently back to the operating room for evacuation of said hematoma. There is also the risk of infection, which can be superficial and can be managed with p.o. antibiotics. However, the patient may develop deeper-seated infection, which may require return to the operating room. Should the infection be in the area of the spinal instrumentation, this will cause a dilemma since there might be a need to remove the spinal instrumentation and/or allograft. There is also the possibility of potential injury to the esophageus, the trachea, and the carotid artery. There is also the risks of stroke on the right cerebral circulation should an undiagnosed plaque be propelled from the right carotid. There is also the possibility hoarseness of the voice secondary to injury to the recurrent laryngeal nerve. There is also the risk of pseudoarthrosis and hardware failure. She understood all of these risks and agreed to have the procedure performed."""
]})))
```

```scala
...

val model = NerDLModel.pretrained("ner_large_clinical","en","clinical/models")
	.setInputCols("sentence","token","word_embeddings")
	.setOutputCol("ner")
    
...

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, word_embeddings, model, ner_converter))

val result = pipeline.fit(Seq.empty["""This is the case of a very pleasant 46-year-old Caucasian female with subarachnoid hemorrhage secondary to ruptured left posteroinferior cerebellar artery aneurysm, which was clipped. The patient last underwent a right frontal ventricular peritoneal shunt on 10/12/07. This resulted in relief of left chest pain, but the patient continued to complaint of persistent pain to the left shoulder and left elbow. She was seen in clinic on 12/11/07 during which time MRI of the left shoulder showed no evidence of rotator cuff tear. She did have a previous MRI of the cervical spine that did show an osteophyte on the left C6-C7 level. Based on this, negative MRI of the shoulder, the patient was recommended to have anterior cervical discectomy with anterior interbody fusion at C6-C7 level. Operation, expected outcome, risks, and benefits were discussed with her. Risks include, but not exclusive of bleeding and infection, bleeding could be soft tissue bleeding, which may compromise airway and may result in return to the operating room emergently for evacuation of said hematoma. There is also the possibility of bleeding into the epidural space, which can compress the spinal cord and result in weakness and numbness of all four extremities as well as impairment of bowel and bladder function. Should this occur, the patient understands that she needs to be brought emergently back to the operating room for evacuation of said hematoma. There is also the risk of infection, which can be superficial and can be managed with p.o. antibiotics. However, the patient may develop deeper-seated infection, which may require return to the operating room. Should the infection be in the area of the spinal instrumentation, this will cause a dilemma since there might be a need to remove the spinal instrumentation and/or allograft. There is also the possibility of potential injury to the esophageus, the trachea, and the carotid artery. There is also the risks of stroke on the right cerebral circulation should an undiagnosed plaque be propelled from the right carotid. There is also the possibility hoarseness of the voice secondary to injury to the recurrent laryngeal nerve. There is also the risk of pseudoarthrosis and hardware failure. She understood all of these risks and agreed to have the procedure performed."""].toDS.toDF("text")).transform(data)
```
</div>

## Results
```bash
+--------------------------------------------------------+---------+
|chunk                                                   |ner_label|
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
|---------------|----------------------------------|
| Name:          | ner_large_clinical               |
| Type:   | NerDLModel                       |
| Compatibility: | Spark NLP 2.5.0+                           |
| License:       | Licensed                         |
| Edition:       | Official                       |
|Input labels:        | [sentence, token, word_embeddings] |
|Output labels:       | [ner]                              |
| Language:      | en                               |
| Dependencies: | embeddings_clinical              |

{:.h2_title}
## Data Source
Trained on data gathered and manually annotated by John Snow Labs
https://www.johnsnowlabs.com/data/

{:.h2_title}
## Benchmarking
```bash
|    | label         |    tp |    fp |    fn |     prec |      rec |       f1 |
|---:|--------------:|------:|------:|------:|---------:|---------:|---------:|
|  0 | I-TREATMENT   |  6625 |  1187 |  1329 | 0.848054 | 0.832914 | 0.840416 |
|  1 | I-PROBLEM     | 15142 |  1976 |  2542 | 0.884566 | 0.856254 | 0.87018  |
|  2 | B-PROBLEM     | 11005 |  1065 |  1587 | 0.911765 | 0.873968 | 0.892466 |
|  3 | I-TEST        |  6748 |   923 |  1264 | 0.879677 | 0.842237 | 0.86055  |
|  4 | B-TEST        |  8196 |   942 |  1029 | 0.896914 | 0.888455 | 0.892665 |
|  5 | B-TREATMENT   |  8271 |  1265 |  1073 | 0.867345 | 0.885167 | 0.876165 |
|  6 | Macro-average | 55987 |  7358 |  8824 | 0.881387 | 0.863166 | 0.872181 |
|  7 | Micro-average | 55987 |  7358 |  8824 | 0.883842 | 0.86385  | 0.873732 |