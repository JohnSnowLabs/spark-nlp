---
layout: model
title: Deidentification NER (Large)
author: John Snow Labs
name: ner_deid_large
date: 2020-03-04
tags: [ner, en, deidentify, licensed]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
Deidentification NER (Large) is a Named Entity Recognition model that annotates text to find protected health information that may need to be deidentified. The entities it annotates are Age, Contact, Date, Id, Location, Name, and Profession. Clinical NER is trained with the 'embeddings_clinical' word embeddings model, so be sure to use the same embeddings in the pipeline.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_DEMOGRAPHICS){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_DEMOGRAPHICS.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_deid_large_en_2.4.2_2.4_1587513305751.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

ner = NerDLModel.pretrained("ner_deid_large", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
```

```scala

val ner = NerDLModel.pretrained("ner_deid_large", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```

</div>

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|ner_deid_large|
|Type:|ner|
|Compatibility:| Spark NLP for Healthcare 2.4.2+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|sentence, token, embeddings|
|Output Labels:|ner|
|Language:|en|
|Case sensitive:|false|


{:.h2_title}
## Source
The model is imported from [https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/](https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/)