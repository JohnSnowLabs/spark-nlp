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
Finds sentence bounds in raw text. Applies a Named Entity Recognition DL model. The Chunk column should be generated via the NER Converter annotator from the outputs of a NER annotator

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
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
  .pretrained("sentence_detector_dl", "en") \
  .setInputCols(["document"]) \
  .setOutputCol("sentences")

sd_model = LightPipeline(PipelineModel(stages=[documenter, sentencerDL]))
sd_model.fullAnnotate("""John loves Mary.Mary loves Peter. Peter loves Helen .Helen loves John; Total: four people involved.""")

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
| Name:          | sentence_detector_dl_healthcare           |
| Type:   | DeepSentenceDetector                      |
| Compatibility: | Spark NLP 2.6.0+                                     |
| License:       | Licensed                                  |
| Edition:       | Official                                |
|Input labels:        | [document, token, chunk_from_ner_converter] |
|Output labels:       | [sentence ]                                 |
| Language:      | en                                        |


{:.h2_title}
## Data Source
Please visit the [repo](https://github.com/dbmdz/deep-eos) for more information
https://github.com/dbmdz/deep-eos