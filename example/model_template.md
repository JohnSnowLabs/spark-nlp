---
layout: model
title: 
author: John Snow Labs
name: 
date: 
tags: [open_source, ner, fr]
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description


{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_FR){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_FR.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](||https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/wikiner_6B_300_fr_2.1.0_2.4_1564817386216.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

ner = NerDLModel.pretrained("wikiner_6B_300", "fr") \
        .setInputCols(["document", "token", "embeddings"]) \
        .setOutputCol("ner")
```

```scala

val ner = NerDLModel.pretrained("wikiner_6B_300", "fr")
        .setInputCols(Array("document", "token", "embeddings"))
        .setOutputCol("ner")
```

</div>

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|Name|
|Type:|ner|
|Compatibility:| Spark NLP 2.5.0|
|License:|Open Source|
|Edition:|Official|
|Spark inputs:|sentence, token, embeddings|
|Spark outputs:|ner|
|Language:|fr|
|Case sensitive:|false|


{:.h2_title}
## Source
The model is imported from [https://fr.wikipedia.org](https://fr.wikipedia.org)