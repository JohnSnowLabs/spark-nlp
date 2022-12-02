---
layout: model
title: Part of Speech for Spanish
author: John Snow Labs
name: pos_ud_gsd
date: 2020-02-17 00:16:00 +0800
task: Part of Speech Tagging
language: es
edition: Spark NLP 2.4.0
spark_version: 2.4
tags: [pos, es]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_es_2.4.0_2.4_1581891015986.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_gsd", "es") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Además de ser el rey del norte, John Snow es un médico inglés y líder en el desarrollo de la anestesia y la higiene médica.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_gsd", "es")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("Además de ser el rey del norte, John Snow es un médico inglés y líder en el desarrollo de la anestesia y la higiene médica.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Además de ser el rey del norte, John Snow es un médico inglés y líder en el desarrollo de la anestesia y la higiene médica."""]
pos_df = nlu.load('es.pos.ud_gsd').predict(text, output_level='token')
pos_df
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=5, result='ADV', metadata={'word': 'Además'}),
Row(annotatorType='pos', begin=7, end=8, result='ADP', metadata={'word': 'de'}),
Row(annotatorType='pos', begin=10, end=12, result='AUX', metadata={'word': 'ser'}),
Row(annotatorType='pos', begin=14, end=15, result='DET', metadata={'word': 'el'}),
Row(annotatorType='pos', begin=17, end=19, result='NOUN', metadata={'word': 'rey'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_gsd|
|Type:|pos|
|Compatibility:|Spark NLP 2.4.0|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|es|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)