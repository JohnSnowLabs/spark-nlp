---
layout: model
title: Deep Sentence Detector Healthcare
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
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sentence_detector_dl_healthcare_en_2.6.0_2.4_1600001082565.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models") \
	.setInputCols(["document"]) \
	.setOutputCol("sentence") 
```

```scala
val model = SentenceDetectorDLModel.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
	.setInputCols(Array("document"))
	.setOutputCol("sentence")
```
</div>



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
