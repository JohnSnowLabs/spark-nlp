---
layout: model
title: Named Entity Recognition in Romanian Official Documents (Small)
author: John Snow Labs
name: legner_romanian_official_sm
date: 2022-11-10
tags: [ro, ner, legal, licensed]
task: Named Entity Recognition
language: ro
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: LegalNerModel
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This is a small version of NER model that extracts only PER(Person), LOC(Location), ORG(Organization) and DATE entities from Romanian Official Documents.

## Predicted Entities

`PER`, `LOC`, `ORG`, `DATE`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/LEGNER_ROMANIAN_OFFICIAL/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_romanian_official_sm_ro_1.0.0_3.0_1668082337617.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_romanian_official_sm_ro_1.0.0_3.0_1668082337617.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")\

sentence_detector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
    .setInputCols(["document"])\
    .setOutputCol("sentence")\

tokenizer = nlp.Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")

embeddings = nlp.BertEmbeddings.pretrained("bert_base_cased", "ro")\
    .setInputCols("sentence", "token")\
    .setOutputCol("embeddings")\
    .setMaxSentenceLength(512)\
    .setCaseSensitive(True)

ner_model = legal.NerModel.pretrained("legner_romanian_official_sm", "ro", "legal/models")\
    .setInputCols(["sentence", "token", "embeddings"])\
    .setOutputCol("ner")\

ner_converter = nlp.NerConverter()\
    .setInputCols(["sentence", "token", "ner"])\
    .setOutputCol("ner_chunk")

pipeline = Pipeline(stages=[
    document_assembler,
    sentence_detector,
    tokenizer,
    embeddings,
    ner_model,
    ner_converter   
    ])

model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

data = spark.createDataFrame([["""Prezentul ordin se publică în Monitorul Oficial al României, Partea I. Ministrul sănătății, Sorina Pintea București, 28 februarie 2019.""]]).toDF("text")
                             
result = model.transform(data)
```

</div>

## Results

```bash
+-----------------------------+-----+
|chunk                        |label|
+-----------------------------+-----+
|Monitorul Oficial al României|ORG  |
|Sorina Pintea                |PER  |
|București                    |LOC  |
|28 februarie 2019            |DATE |
+-----------------------------+-----+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_romanian_official_sm|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|ro|
|Size:|16.4 MB|

## References

Dataset is available [here](https://zenodo.org/record/7025333#.Y2zsquxBx83).

## Benchmarking

```bash
| label        | precision | recall | f1-score | support |
|--------------|-----------|--------|----------|---------|
| DATE         | 0.87      | 0.96   | 0.91     | 397     |
| LOC          | 0.87      | 0.78   | 0.83     | 190     |
| ORG          | 0.90      | 0.93   | 0.91     | 559     |
| PER          | 0.98      | 0.93   | 0.95     | 108     |
| micro-avg    | 0.89      | 0.92   | 0.90     | 1254    |
| macro-avg    | 0.91      | 0.90   | 0.90     | 1254    |
| weighted-avg | 0.89      | 0.92   | 0.90     | 1254    |
```
