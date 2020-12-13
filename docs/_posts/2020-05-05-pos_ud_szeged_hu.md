---
layout: model
title: Part of Speech for Hungarian
author: John Snow Labs
name: pos_ud_szeged
date: 2020-05-05 12:50:00 +0800
tags: [pos, hu]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description
This model annotates the part of speech of tokens in a text. The [parts of speech](https://universaldependencies.org/u/pos/) annotated include PRON (pronoun), CCONJ (coordinating conjunction), and 15 others. The part of speech model is useful for extracting the grammatical structure of a piece of text automatically.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/2da56c087da53a2fac1d51774d49939e05418e57/tutorials/Certification_Trainings/Public/6.Playground_DataFrames.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_szeged_hu_2.5.0_2.4_1588671966774.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python

pos = PerceptronModel.pretrained("pos_ud_szeged", "hu") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Az északi király kivételével John Snow angol orvos, vezető szerepet játszik az érzéstelenítés és az orvosi higiénia fejlesztésében.")
```

```scala

val pos = PerceptronModel.pretrained("pos_ud_szeged", "hu")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
```

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=1, result='DET', metadata={'word': 'Az'}, embeddings=[]),
Row(annotatorType='pos', begin=3, end=8, result='ADJ', metadata={'word': 'északi'}, embeddings=[]),
Row(annotatorType='pos', begin=10, end=15, result='NOUN', metadata={'word': 'király'}, embeddings=[]),
Row(annotatorType='pos', begin=17, end=27, result='NOUN', metadata={'word': 'kivételével'}, embeddings=[]),
Row(annotatorType='pos', begin=29, end=32, result='PROPN', metadata={'word': 'John'}, embeddings=[]),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_szeged|
|Type:|pos|
|Compatibility:|Spark NLP 2.5.0+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|hu|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)