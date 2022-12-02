---
layout: model
title: Part of Speech for Irish
author: John Snow Labs
name: pos_ud_idt
date: 2020-07-29 23:34:00 +0800
task: Part of Speech Tagging
language: ga
edition: Spark NLP 2.5.5
spark_version: 2.4
tags: [pos, ga]
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
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/2da56c087da53a2fac1d51774d49939e05418e57/tutorials/Certification_Trainings/Public/6.Playground_DataFrames.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_idt_ga_2.5.5_2.4_1596054150271.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_idt", "ga") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Seachas a bheith ina rí ar an tuaisceart, is dochtúir Sasanach é John Snow agus ceannaire i bhforbairt ainéistéise agus sláinteachas míochaine.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_idt", "ga")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("Seachas a bheith ina rí ar an tuaisceart, is dochtúir Sasanach é John Snow agus ceannaire i bhforbairt ainéistéise agus sláinteachas míochaine.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Seachas a bheith ina rí ar an tuaisceart, is dochtúir Sasanach é John Snow agus ceannaire i bhforbairt ainéistéise agus sláinteachas míochaine."""]
pos_df = nlu.load('ga.pos').predict(text, output_level='token')
pos_df
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=6, result='ADP', metadata={'word': 'Seachas'}),
Row(annotatorType='pos', begin=8, end=8, result='PART', metadata={'word': 'a'}),
Row(annotatorType='pos', begin=10, end=15, result='NOUN', metadata={'word': 'bheith'}),
Row(annotatorType='pos', begin=17, end=19, result='ADP', metadata={'word': 'ina'}),
Row(annotatorType='pos', begin=21, end=22, result='NOUN', metadata={'word': 'rí'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_idt|
|Type:|pos|
|Compatibility:|Spark NLP 2.5.5+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|ga|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)