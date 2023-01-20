---
layout: model
title: RCT Binary Classifier (USE)
author: John Snow Labs
name: rct_binary_classifier_use
date: 2022-05-27
tags: [licensed, clinical, rct, classifier, en]
task: Text Classification
language: en
edition: Healthcare NLP 3.4.2
spark_version: 3.0
supported: true
annotator: ClassifierDLModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---


## Description


This model is a USE based classifier that can classify if an article is a randomized clinical trial (RCT) or not.


## Predicted Entities


`true`, `false`


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/CLASSIFICATION_RCT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/CLASSIFICATION_RCT.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/rct_binary_classifier_use_en_3.4.2_3.0_1653676810143.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/clinical/models/rct_binary_classifier_use_en_3.4.2_3.0_1653676810143.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}


## How to use


<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = DocumentAssembler()\
        .setInputCol("text")\
        .setOutputCol("document")

use = UniversalSentenceEncoder.pretrained()\
        .setInputCols("document")\
        .setOutputCol("sentence_embeddings")

classifier_dl = ClassifierDLModel.pretrained("rct_binary_classifier_use", "en", "clinical/models")\
        .setInputCols(["sentence_embeddings"])\
        .setOutputCol("class")

use_clf_pipeline = Pipeline(
    stages = [
        document_assembler,
        use,
        classifier_dl
    ])

data = spark.createDataFrame([["""Abstract:Based on the American Society of Anesthesiologists' Practice Guidelines for Sedation and Analgesia by Non-Anesthesiologists (ASA-SED), a sedation training course aimed at improving medical safety was developed by the Japanese Association for Medical Simulation in 2011. This study evaluated the effect of debriefing on participants' perceptions of the essential points of the ASA-SED. A total of 38 novice doctors participated in the sedation training course during the research period. Of these doctors, 18 participated in the debriefing group, and 20 participated in non-debriefing group. Scoring of participants' guideline perceptions was conducted using an evaluation sheet (nine items, 16 points) created based on the ASA-SED. The debriefing group showed a greater perception of the ASA-SED, as reflected in the significantly higher scores on the evaluation sheet (median, 16 points) than the control group (median, 13 points; p < 0.05). No significant differences were identified before or during sedation, but the difference after sedation was significant (p < 0.05). Debriefing after sedation training courses may contribute to better perception of the ASA-SED, and may lead to enhanced attitudes toward medical safety during sedation and analgesia. """]]).toDF("text")

result = use_clf_pipeline.fit(data).transform(data)
```
```scala
val documenter = new DocumentAssembler() 
    .setInputCol("text") 
    .setOutputCol("document")

val use = UniversalSentenceEncoder.pretrained()
    .setInputCols("document")
    .setOutputCol("sentence_embeddings")

val classifier_dl = ClassifierDLModel.pretrained("rct_binary_classifier_use", "en", "clinical/models")
        .setInputCols(Array("sentence_embeddings"))
        .setOutputCol("class")

val use_clf_pipeline = new Pipeline().setStages(Array(documenter, use, classifier_dl))

val data = Seq("""Abstract:Based on the American Society of Anesthesiologists' Practice Guidelines for Sedation and Analgesia by Non-Anesthesiologists (ASA-SED), a sedation training course aimed at improving medical safety was developed by the Japanese Association for Medical Simulation in 2011. This study evaluated the effect of debriefing on participants' perceptions of the essential points of the ASA-SED. A total of 38 novice doctors participated in the sedation training course during the research period. Of these doctors, 18 participated in the debriefing group, and 20 participated in non-debriefing group. Scoring of participants' guideline perceptions was conducted using an evaluation sheet (nine items, 16 points) created based on the ASA-SED. The debriefing group showed a greater perception of the ASA-SED, as reflected in the significantly higher scores on the evaluation sheet (median, 16 points) than the control group (median, 13 points; p < 0.05). No significant differences were identified before or during sedation, but the difference after sedation was significant (p < 0.05). Debriefing after sedation training courses may contribute to better perception of the ASA-SED, and may lead to enhanced attitudes toward medical safety during sedation and analgesia. """).toDS.toDF("text")

val result = use_clf_pipeline.fit(data).transform(data)
```
</div>


## Results


```bash
| text                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      | rct  |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------|
|    Abstract:Based on the American Society of Anesthesiologists' Practice Guidelines for Sedation and Analgesia by Non-Anesthesiologists (ASA-SED), a sedation training course aimed at improving medical safety was developed by the Japanese Association for Medical Simulation in 2011. This study evaluated the effect of debriefing on participants' perceptions of the essential points of the ASA-SED. A total of 38 novice doctors participated in the sedation training course during the research period. Of these doctors, 18 participated in the debriefing group, and 20 participated in non-debriefing group. Scoring of participants' guideline perceptions was conducted using an evaluation sheet (nine items, 16 points) created based on the ASA-SED. The debriefing group showed a greater perception of the ASA-SED, as reflected in the significantly higher scores on the evaluation sheet (median, 16 points) than the control group (median, 13 points; p < 0.05). No significant differences were identified before or during sedation, but the difference after sedation was significant (p < 0.05). Debriefing after sedation training courses may contribute to better perception of the ASA-SED, and may lead to enhanced attitudes toward medical safety during sedation and analgesia.     | true |


```


{:.model-param}
## Model Information


{:.table-model}
|---|---|
|Model Name:|rct_binary_classifier_use|
|Compatibility:|Healthcare NLP 3.4.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence_embeddings]|
|Output Labels:|[class]|
|Language:|en|
|Size:|20.9 MB|


## References


https://arxiv.org/abs/1710.06071


## Benchmarking


```bash
       label  precision    recall  f1-score   support
       false       0.84      0.80      0.82      2915
        true       0.78      0.82      0.80      2545
    accuracy       -         -         0.81      5460
   macro-avg       0.81      0.81      0.81      5460
weighted-avg       0.81      0.81      0.81      5460
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTQ1NjAxOTk1M119
-->