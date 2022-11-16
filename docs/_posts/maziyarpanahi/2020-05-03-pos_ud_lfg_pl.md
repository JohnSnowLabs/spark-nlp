---
layout: model
title: Part of Speech for Polish
author: John Snow Labs
name: pos_ud_lfg
date: 2020-05-03 18:18:00 +0800
task: Part of Speech Tagging
language: pl
edition: Spark NLP 2.5.0
spark_version: 2.4
tags: [pos, pl]
supported: true
annotator: PerceptronModel
article_header:
   type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description
This model annotates the part of speech of tokens in a text. The [parts of speech](https://universaldependencies.org/u/pos/) annotated include PRON (pronoun), CCONJ (coordinating conjunction), and 15 others. The part of speech model is useful for extracting the grammatical structure of a piece of text automatically.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://githubtocolab.com/JohnSnowLabs/spark-nlp-workshop/blob/2da56c087da53a2fac1d51774d49939e05418e57/tutorials/Certification_Trainings/Public/6.Playground_DataFrames.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_lfg_pl_2.5.0_2.4_1588518541171.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_lfg", "pl") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Oprócz bycia królem północy, John Snow jest angielskim lekarzem i liderem w rozwoju anestezjologii i higieny medycznej.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_lfg", "pl")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("Oprócz bycia królem północy, John Snow jest angielskim lekarzem i liderem w rozwoju anestezjologii i higieny medycznej.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Oprócz bycia królem północy, John Snow jest angielskim lekarzem i liderem w rozwoju anestezjologii i higieny medycznej."""]
pos_df = nlu.load('pl.pos.ud_lfg').predict(text, output_level='token')
pos_df
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=5, result='ADP', metadata={'word': 'Oprócz'}),
Row(annotatorType='pos', begin=7, end=11, result='NOUN', metadata={'word': 'bycia'}),
Row(annotatorType='pos', begin=13, end=18, result='NOUN', metadata={'word': 'królem'}),
Row(annotatorType='pos', begin=20, end=26, result='NOUN', metadata={'word': 'północy'}),
Row(annotatorType='pos', begin=27, end=27, result='PUNCT', metadata={'word': ','}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_lfg|
|Type:|pos|
|Compatibility:|Spark NLP 2.5.0+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|pl|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)