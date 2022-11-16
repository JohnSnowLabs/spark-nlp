---
layout: model
title: Part of Speech for Marathi
author: John Snow Labs
name: pos_ud_ufal
date: 2020-07-29 23:34:00 +0800
task: Part of Speech Tagging
language: mr
edition: Spark NLP 2.5.5
spark_version: 2.4
tags: [pos, mr]
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
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_ufal_mr_2.5.5_2.4_1596054314811.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

{:.h2_title}
## How to use 

<div class="tabs-box" markdown="1">

{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
pos = PerceptronModel.pretrained("pos_ud_ufal", "mr") \
    .setInputCols(["document", "token"]) \
    .setOutputCol("pos")
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, pos])
light_pipeline = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))
results = light_pipeline.fullAnnotate("उत्तरेचा राजा होण्याव्यतिरिक्त, जॉन स्नो एक इंग्रज चिकित्सक आहे आणि भूल आणि वैद्यकीय स्वच्छतेच्या विकासासाठी अग्रगण्य आहे.")
```

```scala
...
val pos = PerceptronModel.pretrained("pos_ud_ufal", "mr")
    .setInputCols(Array("document", "token"))
    .setOutputCol("pos")
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, pos))
val data = Seq("उत्तरेचा राजा होण्याव्यतिरिक्त, जॉन स्नो एक इंग्रज चिकित्सक आहे आणि भूल आणि वैद्यकीय स्वच्छतेच्या विकासासाठी अग्रगण्य आहे.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu

text = ["""उत्तरेचा राजा होण्याव्यतिरिक्त, जॉन स्नो एक इंग्रज चिकित्सक आहे आणि भूल आणि वैद्यकीय स्वच्छतेच्या विकासासाठी अग्रगण्य आहे."""]
pos_df = nlu.load('mr.pos').predict(text)
pos_df
```

</div>

{:.h2_title}
## Results

```bash
[Row(annotatorType='pos', begin=0, end=7, result='NOUN', metadata={'word': 'उत्तरेचा'}),
Row(annotatorType='pos', begin=9, end=12, result='NOUN', metadata={'word': 'राजा'}),
Row(annotatorType='pos', begin=14, end=29, result='NOUN', metadata={'word': 'होण्याव्यतिरिक्त'}),
Row(annotatorType='pos', begin=30, end=30, result='PUNCT', metadata={'word': ','}),
Row(annotatorType='pos', begin=32, end=34, result='NOUN', metadata={'word': 'जॉन'}),
...]
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_ufal|
|Type:|pos|
|Compatibility:|Spark NLP 2.5.5+|
|Edition:|Official|
|Input labels:|[token]|
|Output labels:|[pos]|
|Language:|mr|
|Case sensitive:|false|
|License:|Open Source|

{:.h2_title}
## Data Source
The model is imported from [https://universaldependencies.org](https://universaldependencies.org)