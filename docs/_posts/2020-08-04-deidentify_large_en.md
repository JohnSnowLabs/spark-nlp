---
layout: model
title: Deidentify (Large)
author: John Snow Labs
name: deidentify_large
date: 2020-08-04
tags: [deid, en, licensed]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
Deidentify (Large) is a deidentification model. It identifies instances of protected health information in text documents, and it can either obfuscate them (e.g., replacing names with different, fake names) or mask them (e.g., replacing "2020-06-04" with "<DATE>"). This model is useful for maintaining HIPAA compliance when dealing with text documents that contain protected health information.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/DEID_PHI_TEXT){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/DEID_PHI_TEXT.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](||https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/nerdl_deid_en_1.8.0_2.4_1545462443516.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

deid = DeIdentificationModel.pretrained("deidentify_large", "en") \
        .setInputCols(["sentence", "token", "ner_chunk"]) \
        .setOutputCol("obfuscated") \
          .setMode("obfuscate")
```

```scala

val deid = DeIdentificationModel.pretrained("deidentify_large", "en")
        .setInputCols(Array("sentence", "token", "ner_chunk"))
        .setOutputCol("obfuscated") \
          .setMode("obfuscate")
```

</div>

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|deidentify_large|
|Type:|deid|
|Compatibility:| Spark NLP JSL2.5.5|
|License:|Licensed|
|Edition:|Official|
|Spark inputs:|sentence, token, ner_chunk|
|Spark outputs:|obfuscated|
|Language:|en|
|Case sensitive:|false|


{:.h2_title}
## Source
The model is imported from [https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/](https://portal.dbmi.hms.harvard.edu/projects/n2c2-2014/)