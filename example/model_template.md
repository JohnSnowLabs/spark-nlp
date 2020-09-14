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

{:.h2_title}
## Results
The output is a dataframe with a Relation column and a Confidence column....


{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|Name|
|Type:|ner|
|Compatibility:| Spark NLP 2.5.0|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|sentence, token, embeddings|
|Output Labels:|ner|
|Language:|fr|
|Case sensitive:|false|


{:.h2_title}
## Source
The model is trained based on data from ...

{:.h2_title}
## Dataset used for training
Trained on ...
