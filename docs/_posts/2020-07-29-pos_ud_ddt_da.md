---
layout: model
title: Part of Speech for Danish
author: John Snow Labs
name: pos_ud_ddt
date: 2020-07-29 23:34:00 +0800
tags: [pos, da]
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
This model annotates the part of speech of tokens in a text. The [parts of speech](https://universaldependencies.org/u/pos/) annotated include PRON (pronoun), CCONJ (coordinating conjunction), and 15 others. The part of speech model is useful for extracting the grammatical structure of a piece of text automatically.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/2da56c087da53a2fac1d51774d49939e05418e57/tutorials/Certification_Trainings/Public/6.Playground_DataFrames.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_ddt_da_2.5.5_2.4_1596053892919.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_ddt", "da") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("John Snow er bortset fra at være kongen i nord, en engelsk læge og en leder inden for udvikling af anæstesi og medicinsk hygiejne.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_ddt", "da")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val result = pipeline.fit(Seq.empty["John Snow er bortset fra at være kongen i nord, en engelsk læge og en leder inden for udvikling af anæstesi og medicinsk hygiejne."].toDS.toDF("text")).transform(data)
```

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=3, result='PROPN', metadata={'word': 'John'}),
Row(annotatorType='pos', begin=5, end=8, result='PROPN', metadata={'word': 'Snow'}),
Row(annotatorType='pos', begin=10, end=11, result='AUX', metadata={'word': 'er'}),
Row(annotatorType='pos', begin=13, end=19, result='VERB', metadata={'word': 'bortset'}),
Row(annotatorType='pos', begin=21, end=23, result='ADP', metadata={'word': 'fra'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_ddt|
|Type:|pos|
|Compatibility:|Spark NLP 2.5.5+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|da|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)