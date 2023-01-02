---
layout: model
title: Named Entity Recognition in Romanian Official Documents (Medium)
author: John Snow Labs
name: legner_romanian_official_md
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

This is a medium version of NER model that extracts PER(Person), LOC(Location), ORG(Organization), DATE and LEGAL entities from Romanian Official Documents. Different from small version, it labels all entities related to legal domain as LEGAL.

## Predicted Entities

`PER`, `LOC`, `ORG`, `DATE`, `LEGAL`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/LEGNER_ROMANIAN_OFFICIAL/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_romanian_official_md_ro_1.0.0_3.0_1668083301892.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

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

ner_model = legal.NerModel.pretrained("legner_romanian_official_md", "ro", "legal/models")\
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

data = spark.createDataFrame([["""Anexa nr. 1 la Ordinul ministrului sănătății nr. 1.468 / 2018 pentru aprobarea prețurilor maximale ale medicamentelor de uz uman, valabile în România, care pot fi utilizate / comercializate de către deținătorii de autorizație de punere pe piață a medicamentelor sau reprezentanții acestora, distribuitorii angro și furnizorii de servicii medicale și medicamente pentru acele medicamente care fac obiectul unei relații contractuale cu Ministerul Sănătății, casele de asigurări de sănătate și / sau direcțiile de sănătate publică județene și a municipiului București, cuprinse în Catalogul național al prețurilor medicamentelor autorizate de punere pe piață în România, a prețurilor de referință generice și a prețurilor de referință inovative, publicat în Monitorul Oficial al României, Partea I nr. 989 și 989 bis din 22 noiembrie 2018, cu modificările și completările ulterioare, se modifică și se completează conform anexei care face parte integrantă din prezentul ordin."""]]).toDF("text")
                             
result = model.transform(data)
```

</div>

## Results

```bash
+----------------------------------------------+-----+
|chunk                                         |label|
+----------------------------------------------+-----+
|Ordinul ministrului sănătății nr. 1.468 / 2018|LEGAL|
|România                                       |LOC  |
|Ministerul Sănătății                          |ORG  |
|București                                     |LOC  |
|România                                       |LOC  |
|Monitorul Oficial al României                 |ORG  |
|22 noiembrie 2018                             |DATE |
+----------------------------------------------+-----+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_romanian_official_md|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|ro|
|Size:|16.5 MB|

## References

Dataset is available [here](https://zenodo.org/record/7025333#.Y2zsquxBx83).

## Benchmarking

```bash
label         precision  recall  f1-score  support
DATE          0.84       0.92    0.88      218
LEGAL         0.89       0.96    0.92      337
LOC           0.82       0.77    0.79      158
ORG           0.87       0.88    0.88      463
PER           0.97       0.97    0.97      87
micro-avg     0.87       0.90    0.89      1263
macro-avg     0.88       0.90    0.89      1263
weighted-avg  0.87       0.90    0.89      1263
```
