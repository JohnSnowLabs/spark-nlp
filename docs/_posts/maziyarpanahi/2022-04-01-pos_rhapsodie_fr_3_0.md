---
layout: model
title: Part of Speech UD model for French (pos_rhapsodie)
author: John Snow Labs
name: pos_rhapsodie
date: 2022-04-01
tags: [open_source, universal_dependency, pos, part_of_speech, fr, french]
task: Part of Speech Tagging
language: fr
edition: Spark NLP 3.4.3
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Pretrained Part of Speech model (`pos_rhapsodie`) trained on Universal Dependencies 2.9 (UD_French-Rhapsodie) in French language.

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/GRAMMAR_EN.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_rhapsodie_fr_3.4.3_3.0_1648798260839.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use

This sample snippet may not include all the required components of the pipeline for readability purposes. However, you can find a complete example of all the end-to-end components of the pipeline by clicking the "Open in Colab" link included above.




<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
          
document = DocumentAssembler()\ 
    .setInputCol("text")\ 
    .setOutputCol("document")

sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\ 
    .setInputCols(["document"])\ 
    .setOutputCol("sentence")

tokenizer = Tokenizer()\ 
    .setInputCols(["sentence"])\ 
    .setOutputCol("token") 

pos = PerceptronModel.pretrained("pos_rhapsodie", "fr")\ 
    .setInputCols(["sentence", "token"])\ 
    .setOutputCol("pos")
    
pipeline = Pipeline(stages=[document, sentence, tokenizer, pos])
    
data = spark.createDataFrame([["I love Spark NLP"]]).toDF("text")

result = pipeline.fit(data).transform(data)

```
```scala

val document = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentence = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
  .setInputCols("document")
  .setOutputCol("sentence")

val tokenizer = new Tokenizer() 
    .setInputCols("sentence") 
    .setOutputCol("token")
    
val pos = PerceptronModel.pretrained("pos_rhapsodie", "fr")
    .setInputCols("sentence", "token")
    .setOutputCol("pos")
    
val pipeline = new Pipeline().setStages(Array(document, sentence, tokenizer, pos))

val data = Seq("I love Spark NLP").toDF("text")

val result = pipeline.fit(data).transform(data)
```
</div>

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_rhapsodie|
|Compatibility:|Spark NLP 3.4.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, form]|
|Output Labels:|[pos]|
|Language:|fr|
|Size:|328.3 KB|

## References

Model is trained on Universal Dependencies (treebank 2.9) `UD_French-Rhapsodie`

[https://github.com/UniversalDependencies/UD_French-Rhapsodie](https://github.com/UniversalDependencies/UD_French-Rhapsodie)