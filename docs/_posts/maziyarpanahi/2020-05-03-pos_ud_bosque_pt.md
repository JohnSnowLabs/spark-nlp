---
layout: model
title: Part of Speech for Portuguese
author: John Snow Labs
name: pos_ud_bosque
date: 2020-05-03 12:54:00 +0800
task: Part of Speech Tagging
language: pt
edition: Spark NLP 2.5.0
spark_version: 2.4
tags: [pos, pt]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_bosque_pt_2.5.0_2.4_1588499443093.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_bosque", "pt") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("Além de ser o rei do norte, John Snow é um médico inglês e líder no desenvolvimento de anestesia e higiene médica.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_bosque", "pt")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("Annet enn å være kongen i nord, er John Snow en engelsk lege og en leder innen utvikling av anestesi og medisinsk hygiene.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""Além de ser o rei do norte, John Snow é um médico inglês e líder no desenvolvimento de anestesia e higiene médica."""]
pos_df = nlu.load('pt.pos.ud_bosque').predict(text, output_level='token')
pos_df
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=3, result='ADV', metadata={'word': 'Além'}),
Row(annotatorType='pos', begin=5, end=6, result='ADP', metadata={'word': 'de'}),
Row(annotatorType='pos', begin=8, end=10, result='AUX', metadata={'word': 'ser'}),
Row(annotatorType='pos', begin=12, end=12, result='DET', metadata={'word': 'o'}),
Row(annotatorType='pos', begin=14, end=16, result='NOUN', metadata={'word': 'rei'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_bosque|
|Type:|pos|
|Compatibility:|Spark NLP 2.5.0+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|pt|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)