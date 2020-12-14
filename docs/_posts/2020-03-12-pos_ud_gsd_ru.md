---
layout: model
title: Part of Speech for Russian
author: John Snow Labs
name: pos_ud_gsd
date: 2020-03-12 13:48:00 +0800
tags: [pos, ru]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_ru_2.4.4_2.4_1584013495069.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_gsd", "ru") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Помимо того, что он король севера, Джон Сноу - английский врач и лидер в разработке анестезии и медицинской гигиены.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_gsd", "ru")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val result = pipeline.fit(Seq.empty["Помимо того, что он король севера, Джон Сноу - английский врач и лидер в разработке анестезии и медицинской гигиены."].toDS.toDF("text")).transform(data)
```
</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=5, result='ADP', metadata={'word': 'Помимо'}),
Row(annotatorType='pos', begin=7, end=10, result='PRON', metadata={'word': 'того'}),
Row(annotatorType='pos', begin=11, end=11, result='PUNCT', metadata={'word': ','}),
Row(annotatorType='pos', begin=13, end=15, result='SCONJ', metadata={'word': 'что'}),
Row(annotatorType='pos', begin=17, end=18, result='PRON', metadata={'word': 'он'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_gsd|
|Type:|pos|
|Compatibility:|Spark NLP 2.4.4|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|ru|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)