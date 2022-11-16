---
layout: model
title: Part of Speech for Galician
author: John Snow Labs
name: pos_ud_treegal
date: 2020-07-29 23:34:00 +0800
task: Part of Speech Tagging
language: gl
edition: Spark NLP 2.5.5
spark_version: 2.4
tags: [pos, gl]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_treegal_gl_2.5.5_2.4_1596053906222.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_treegal", "gl") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Ademais de ser o rei do norte, John Snow é un médico inglés e un líder no desenvolvemento da anestesia e a hixiene médica.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_treegal", "gl")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("Ademais de ser o rei do norte, John Snow é un médico inglés e un líder no desenvolvemento da anestesia e a hixiene médica.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Ademais de ser o rei do norte, John Snow é un médico inglés e un líder no desenvolvemento da anestesia e a hixiene médica."""]
pos_df = nlu.load('gl.pos').predict(text, output_level='token')
pos_df
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=6, result='ADV', metadata={'word': 'Ademais'}),
Row(annotatorType='pos', begin=8, end=9, result='ADP', metadata={'word': 'de'}),
Row(annotatorType='pos', begin=11, end=13, result='AUX', metadata={'word': 'ser'}),
Row(annotatorType='pos', begin=15, end=15, result='DET', metadata={'word': 'o'}),
Row(annotatorType='pos', begin=17, end=19, result='NOUN', metadata={'word': 'rei'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_treegal|
|Type:|pos|
|Compatibility:|Spark NLP 2.5.5+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|gl|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)