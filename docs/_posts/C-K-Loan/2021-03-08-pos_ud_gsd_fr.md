---
layout: model
title: Pat of Speech for French
author: John Snow Labs
name: pos_ud_gsd
date: 2021-03-08
tags: [part_of_speech, open_source, french, pos_ud_gsd, fr]
task: Part of Speech Tagging
language: fr
edition: Spark NLP 3.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Part of speech tagger. Predicts the following tags: 
- DET
- NOUN
- ADJ
- AUX
- VERB
- ADV
- ADP
- SCONJ
- PRON
- PUNCT
- PROPN
- CCONJ
- NUM
- SYM
- X
- PART
- INTJ

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/pos_ud_gsd_fr_3.0.0_3.0_1615211640304.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python

document_assembler = DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

sentence_detector = SentenceDetector()
  .setInputCols(["document"])
  .setOutputCol("sentence")

pos = PerceptronModel.pretrained("pos_ud_gsd", "fr")
  .setInputCols(["document", "token"])
  .setOutputCol("pos")

pipeline = Pipeline(stages=[
  document_assembler,
  sentence_detector,
  posTagger
])

example = spark.createDataFrame(pd.DataFrame({'text': ["Bonjour de John Snow Labs! "]}))

result = pipeline.fit(example).transform(example)


```
```scala

val document_assembler = DocumentAssembler()
        .setInputCol("text")
        .setOutputCol("document")

val sentence_detector = SentenceDetector()
        .setInputCols(["document"])
.setOutputCol("sentence")

val pos = PerceptronModel.pretrained("pos_ud_gsd", "fr")
        .setInputCols(Array("document", "token"))
        .setOutputCol("pos")

val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, pos))

val result = pipeline.fit(Seq.empty["Bonjour de John Snow Labs! "].toDS.toDF("text")).transform(data)

```

{:.nlu-block}
```python

import nlu
text = [""Bonjour de John Snow Labs! ""]
token_df = nlu.load('fr.pos.ud_gsd').predict(text)
token_df
    
```
</div>

## Results

```bash
     token    pos
                 
0  Bonjour   INTJ
1       de    ADP
2     John  PROPN
3     Snow  PROPN
4     Labs  PROPN
5        !  PUNCT
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|pos_ud_gsd|
|Compatibility:|Spark NLP 3.0.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[pos]|
|Language:|fr|