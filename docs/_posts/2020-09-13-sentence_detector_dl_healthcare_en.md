---
layout: model
title: Deep Sentence Detector Healthcare
author: John Snow Labs
name: sentence_detector_dl_healthcare
class: DeepSentenceDetector
language: en
repository: clinical/models
date: 2020-09-13
tags: [clinical,sentence_detection,dl,en]
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
Finds sentence bounds in raw text. Applies a Named Entity Recognition DL model. The Chunk column should be generated via the NER Converter annotator from the outputs of a NER annotator  


[https://github.com/dbmdz/deep-eos](https://github.com/dbmdz/deep-eos)

{:.h2_title}
## Data Source
Please visit the [repo](https://github.com/dbmdz/deep-eos) for more information

{:.btn-box}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/sentence_detector_dl_healthcare_en_2.6.0_2.4_1600001082565.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
{:.h2_title}
## How to use 
<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
model = DeepSentenceDetector.pretrained("sentence_detector_dl_healthcare","en","clinical/models")\
	.setInputCols("document","token","chunk_from_ner_converter")\
	.setOutputCol("sentence")
```

```scala
val model = DeepSentenceDetector.pretrained("sentence_detector_dl_healthcare","en","clinical/models")
	.setInputCols("document","token","chunk_from_ner_converter")
	.setOutputCol("sentence")
```
</div>



{:.model-param}
## Model Information

{:.table-model}
|---------------|-------------------------------------------|
| name          | sentence_detector_dl_healthcare           |
| model_class   | DeepSentenceDetector                      |
| compatibility | 2.6.0                                     |
| license       | Licensed                                  |
| edition       | Healthcare                                |
| inputs        | document, token, chunk_from_ner_converter |
| output        | sentence                                  |
| language      | en                                        |

