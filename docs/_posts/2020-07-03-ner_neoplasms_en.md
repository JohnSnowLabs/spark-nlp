---
layout: model
title: Neoplasms NER
author: John Snow Labs
name: ner_neoplasms
date: 2020-07-03
tags: [ner, en, tumor]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
Neoplasms NER is a Named Entity Recognition model that annotates text to find references to tumors. The only entity it annotates is MalignantNeoplasm. Neoplasms NER is trained with the 'embeddings_clinical' word embeddings model, so be sure to use the same embeddings in the pipeline.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/healthcare/NER_TUMOR){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/healthcare/NER_TUMOR.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](||https://s3.amazonaws.com/auxdata.johnsnowlabs.com/clinical/models/ner_neoplasms_es_2.5.3_2.4_1594168624415.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

ner = NerDLModel.pretrained("ner_neoplasms", "en") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
```

```scala

val ner = NerDLModel.pretrained("ner_neoplasms", "en")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```

</div>

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|ner_neoplasms|
|Type:|ner|
|Compatibility:| Spark NLP JSL2.5.3|
|License:|Licensed|
|Edition:|Official|
|Spark inputs:|sentence, token, embeddings|
|Spark outputs:|ner|
|Language:|en|
|Case sensitive:|false|


{:.h2_title}
## Source
The model is imported from [https://temu.bsc.es/cantemist/](https://temu.bsc.es/cantemist/)