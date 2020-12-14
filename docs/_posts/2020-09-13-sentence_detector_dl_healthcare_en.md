---
layout: model
title: Split Sentences in Healthcare Texts
author: John Snow Labs
name: sentence_detector_dl_healthcare
class: DeepSentenceDetector
language: en
repository: clinical/models
date: 2020-09-13
tags: [clinical,sentence_detection,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
SentenceDetectorDL (SDDL) is based on a general-purpose neural network model for sentence boundary detection. The task of sentence boundary detection is to identify sentences within a text. Many natural language processing tasks take a sentence as an input unit, such as part-of-speech tagging, dependency parsing, named entity recognition or machine translation.

In this model, we treated the sentence boundary detection task as a classification problem based on a paper {Deep-EOS: General-Purpose Neural Networks for Sentence Boundary Detection (2020, Stefan Schweter, Sajawel Ahmed) using CNN architecture. We also modified the original implemenation a little bit to cover broken sentences and some impossible end of line chars.


{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/9.SentenceDetectorDL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}{:target="_blank"}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sentence_detector_dl_healthcare_en_2.6.0_2.4_1600001082565.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
documenter = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
sentencerDL = SentenceDetectorDLModel\
  .pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")
sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""John loves Mary.Mary loves Peter. Peter loves Helen .Helen loves John; Total: four people involved.""")
```

```scala
val documenter = DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")
val pipeline = new Pipeline().setStages(Array(documenter, model))
val result = pipeline.fit(Seq.empty["John loves Mary.Mary loves Peter. Peter loves Helen .Helen loves John; Total: four people involved."].toDS.toDF("text")).transform(data)
```
</div>

{:.h2_title}
## Results

```bash
+---+------------------------------+
| 0 | John loves Mary.             |
+---+------------------------------+
| 1 | Mary loves Peter             |
+---+------------------------------+
| 2 | Peter loves Helen .          |
+---+------------------------------+
| 3 | Helen loves John;            |
+---+------------------------------+
| 4 | Total: four people involved. |
+---+------------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---------------|-------------------------------------------|
| Name:          | sentence_detector_dl_healthcare           |
| Type:   | DeepSentenceDetector                      |
| Compatibility: | Spark NLP 2.6.0+                                     |
| License:       | Licensed                                  |
| Edition:       | Official                                |
|Input labels:        | [document] |
|Output labels:       | sentence                                 |
| Language:      | en                                        |


{:.h2_title}
## Data Source
Healthcare SDDL model is trained on domain (healthcare) specific text, annotated internally, to generalize further on clinical notes.

{:.h2_title}
## Benchmarking
```bash
|    | Accuracy | Recall   | Prec     | F1   |
|---:|---------:|---------:|---------:|-----:|
|  0 | 0.98     | 1.00     | 0.96     | 0.98 |
```