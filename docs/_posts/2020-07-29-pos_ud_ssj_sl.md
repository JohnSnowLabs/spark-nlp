---
layout: model
title: Part of Speech for Slovenian
author: John Snow Labs
name: pos_ud_ssj
date: 2020-07-29 23:35:00 +0800
tags: [pos, sl]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_ssj_sl_2.5.5_2.4_1596054388189.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_ssj", "sl") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("John Snow je poleg tega, da je severni kralj, angleški zdravnik in vodilni v razvoju anestezije in zdravstvene higiene.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_ssj", "sl")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val result = pipeline.fit(Seq.empty["John Snow je poleg tega, da je severni kralj, angleški zdravnik in vodilni v razvoju anestezije in zdravstvene higiene."].toDS.toDF("text")).transform(data)
```

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=3, result='PROPN', metadata={'word': 'John'}),
Row(annotatorType='pos', begin=5, end=8, result='PROPN', metadata={'word': 'Snow'}),
Row(annotatorType='pos', begin=10, end=11, result='AUX', metadata={'word': 'je'}),
Row(annotatorType='pos', begin=13, end=17, result='ADP', metadata={'word': 'poleg'}),
Row(annotatorType='pos', begin=19, end=22, result='DET', metadata={'word': 'tega'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_ssj|
|Type:|pos|
|Compatibility:|Spark NLP 2.5.5+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|sl|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)