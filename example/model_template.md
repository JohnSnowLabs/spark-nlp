---
layout: model
title: title of the models (SEO optimized)
author: John Snow Labs
name: name of the model (ex. ner_dl)
date: publishing date
tags: [open_source or licensed, ner, fr, healthcare?] 
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
Short description of the model
### Predicted Labels
The list of labels predicted by the model. Please add explanations if necessary but keep them short...

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
The actual outputs of the code above + some explanations if necessary....

{:.result_box}
```python
root
 |-- text: string (nullable = true)
 |-- document: array (nullable = true)
 |    |-- element: struct (containsNull = true)
 |    |    |-- annotatorType: string (nullable = true)
 |    |    |-- begin: integer (nullable = false)
 |    |    |-- end: integer (nullable = false)
 |    |    |-- result: string (nullable = true)
 |    |    |-- metadata: map (nullable = true)
 |    |    |    |-- key: string
 |    |    |    |-- value: string (valueContainsNull = true)
 |    |    |-- embeddings: array (nullable = true)
 |    |    |    |-- element: float (containsNull = false)
 |-- sentence: array (nullable = true)
 ...
 ```

{:.model-param}
## Model Parameters

{:.table-model}
|---|---|
|Model Name:|Name of the model - same as in the metadata (header of this file)|
|Type:|ner|
|Compatibility:| Spark NLP 2.5.0 + or Spark NLP for Healthcare 2.6.0 +|
|License:|Open Source or Licensed|
|Edition:|Official or Community|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|fr|
|Case sensitive:|false|


{:.h2_title}
## Data Source
The model is trained based on data from ...

{:.h2_title}
## Benchmarking 
Info about accuracy etc...
