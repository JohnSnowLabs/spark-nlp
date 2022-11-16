---
layout: model
title: Detect Restaurant-related Terminology
author: John Snow Labs
name: nerdl_restaurant_100d
date: 2021-12-31
tags: [ner, restaurant, en, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.3.4
spark_version: 3.0
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained with `glove_100d` embeddings to detect restaurant-related terminology.

## Predicted Entities

`Location`, `Cuisine`, `Amenity`, `Restaurant_Name`, `Dish`, `Rating`, `Hours`, `Price`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_RESTAURANT/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_RESTAURANT.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerdl_restaurant_100d_en_3.3.4_3.0_1640949258750.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...

embeddings = WordEmbeddingsModel.pretrained("glove_100d") \
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

nerdl = NerDLModel.pretrained("nerdl_restaurant_100d")\
.setInputCols(["sentence", "token", "embeddings"])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence", "token", "ner"])\
.setOutputCol("ner_chunk")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, nerdl, ner_converter])

text = """Hong Kong’s favourite pasta bar also offers one of the most reasonably priced lunch sets in town! With locations spread out all over the territory Sha Tin – Pici’s formidable lunch menu reads like a highlight reel of the restaurant. Choose from starters like the burrata and arugula salad or freshly tossed tuna tartare, and reliable handmade pasta dishes like pappardelle. Finally, round out your effortless Italian meal with a tidy one-pot tiramisu, of course, an espresso to power you through the rest of the day."""

data = spark.createDataFrame([[text]]).toDF("text")

result = nlp_pipeline.fit(data).transform(data)
```
```scala
...

val embeddings = WordEmbeddingsModel.pretrained("glove_100d")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val nerdl = NerDLModel.pretrained("nerdl_restaurant_100d")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("ner")

val ner_converter = NerConverter()
.setInputCols(Array("sentence", "token", "ner"))
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, nerdl, ner_converter))

val data = Seq("Hong Kong’s favourite pasta bar also offers one of the most reasonably priced lunch sets in town! With locations spread out all over the territory Sha Tin – Pici’s formidable lunch menu reads like a highlight reel of the restaurant. Choose from starters like the burrata and arugula salad or freshly tossed tuna tartare, and reliable handmade pasta dishes like pappardelle. Finally, round out your effortless Italian meal with a tidy one-pot tiramisu, of course, an espresso to power you through the rest of the day.").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.ner.restaurant").predict("""Hong Kong’s favourite pasta bar also offers one of the most reasonably priced lunch sets in town! With locations spread out all over the territory Sha Tin – Pici’s formidable lunch menu reads like a highlight reel of the restaurant. Choose from starters like the burrata and arugula salad or freshly tossed tuna tartare, and reliable handmade pasta dishes like pappardelle. Finally, round out your effortless Italian meal with a tidy one-pot tiramisu, of course, an espresso to power you through the rest of the day.""")
```

</div>

## Results

```bash
+---------------------------+---------------+
|chunk                      |ner_label      |
+---------------------------+---------------+
|favourite                  |Rating         |
|pasta bar                  |Dish           |
|most reasonably            |Price          |
|lunch                      |Hours          |
|in town!                   |Location       |
|Sha Tin – Pici’s           |Restaurant_Name|
|burrata                    |Dish           |
|arugula salad              |Dish           |
|freshly tossed tuna tartare|Dish           |
|reliable                   |Price          |
|handmade pasta             |Dish           |
|pappardelle                |Dish           |
|effortless                 |Amenity        |
|Italian                    |Cuisine        |
|tidy one-pot               |Amenity        |
|espresso                   |Dish           |
+---------------------------+---------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerdl_restaurant_100d|
|Type:|ner|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|
|Size:|14.2 MB|

## Data Source

[https://groups.csail.mit.edu/sls/downloads/restaurant/](https://groups.csail.mit.edu/sls/downloads/restaurant/)

## Benchmarking

```bash
label  precision    recall  f1-score   support
B-Amenity       0.77      0.75      0.76       545
B-Cuisine       0.86      0.88      0.87       524
B-Dish       0.84      0.80      0.82       303
B-Hours       0.67      0.72      0.69       197
B-Location       0.89      0.89      0.89       807
B-Price       0.86      0.87      0.86       169
B-Rating       0.87      0.79      0.83       221
B-Restaurant_Name       0.91      0.94      0.92       388
I-Amenity       0.80      0.75      0.77       561
I-Cuisine       0.71      0.71      0.71       135
I-Dish       0.66      0.77      0.71       104
I-Hours       0.87      0.84      0.86       306
I-Location       0.92      0.87      0.89       834
I-Price       0.64      0.81      0.71        52
I-Rating       0.80      0.85      0.82       118
I-Restaurant_Name       0.82      0.89      0.85       359
O       0.95      0.96      0.96      8634
accuracy        -          -       0.91     14257
macro-avg       0.81      0.83      0.82     14257
weighted-avg       0.91      0.91      0.91     14257
```
