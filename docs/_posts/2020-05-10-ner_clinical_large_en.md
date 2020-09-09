---
layout: model
title: Clinical NER (Large)
author: John Snow Labs
name: ner_clinical_large
date: 2020-05-10
tags: [ner, en]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
Clinical NER (Large) is a Named Entity Recognition model that annotates text to find references to clinical events. The entities it annotates are Problem, Treatment, and Test. Clinical NER is trained with the 'embeddings_clinical' word embeddings model, so be sure to use the same embeddings in the pipeline.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_EVENTS_CLINICAL){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_EVENTS_CLINICAL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](||https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_large_clinical_en_2.5.0_2.4_1590021302624.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use 

{% include programmingLanguageSelectScalaPython.html %}

```python

ner = NerDLModel.pretrained("ner_clinical_large", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
```

```scala

val ner = NerDLModel.pretrained("ner_clinical_large", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|ner_clinical_large|
|Type:|ner|
|Compatibility:| Spark NLP JSL2.5.0|
|Edition:|Official|
|Spark inputs:|sentence, token, embeddings|
|Spark outputs:|ner|
|Language:|en|

|Case sensitive:|false|
|License:|Enterprise|

{:.h2_title}
## Source
The model is imported from [https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)