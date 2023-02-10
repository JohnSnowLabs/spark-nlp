---
layout: model
title: Spanish NER for Laws and Treaties/Agreements (Roberta)
author: John Snow Labs
name: legner_laws_treaties
date: 2022-09-28
tags: [es, legal, ner, laws, treaties, agreements, licensed]
task: Named Entity Recognition
language: es
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
annotator: RoBertaForTokenClassification
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Legal Roberta Named Entity Recognition model in Spanish, able to recognize the following entities:
- LEY: Law
- TRAT_INTL: International Treaty (Agreement)

This model originally trained on scjn dataset, available [here](https://huggingface.co/datasets/scjnugacj/scjn_dataset_ner) and finetuned on scrapped documents (as, for example, [this one](https://www.wipo.int/export/sites/www/pct/es/texts/pdf/pct.pdf)), improving the coverage of the original version, published [here](https://huggingface.co/datasets/scjnugacj/scjn_dataset_ner).

## Predicted Entities

`LAW`, `TRAT_INTL`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_laws_treaties_es_1.0.0_3.0_1664362398391.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legner_laws_treaties_es_1.0.0_3.0_1664362398391.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler() \
       .setInputCol("text") \
       .setOutputCol("document")

sentenceDetector = nlp.SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = nlp.Tokenizer() \
    .setInputCols("sentence") \
    .setOutputCol("token")

tokenClassifier = nlp.RoBertaForTokenClassification.pretrained("legner_laws_treaties","es", "legal/models") \
    .setInputCols(["sentence", "token"]) \
    .setOutputCol("ner")

pipeline = Pipeline(
    stages=[documentAssembler, 
            sentenceDetector, 
            tokenizer, 
            tokenClassifier])

text = "Sin perjuicio de lo dispuesto en el párrafo b), los requisitos y los efectos de una reivindicación de prioridad presentada conforme al párrafo 1), serán los establecidos en el Artículo 4 del Acta de Estocolmo del Convenio de París para la Protección de la Propiedad Industrial."

data = spark.createDataFrame([[""]]).toDF("text")

fitmodel = pipeline.fit(data)

light_model = LightPipeline(fitmodel)

light_result = light_model.fullAnnotate(text)

chunks = []
entities = []

for n in light_result[0]['ner_chunk']:       
    print("{n.result} ({n.metadata['entity']}))
```

</div>

## Results

```bash
para la Protección de la Propiedad Industrial. (TRAT_INTL)
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_laws_treaties|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[ner]|
|Language:|es|
|Size:|464.4 MB|
|Case sensitive:|true|
|Max sentence length:|128|

## References

This model was originally trained on scjn dataset, available [here](https://huggingface.co/datasets/scjnugacj/scjn_dataset_ner) and finetuned on scrapped documents (as, for example, [this one](https://www.wipo.int/export/sites/www/pct/es/texts/pdf/pct.pdf)), improving the coverage of the original version, published [here](https://huggingface.co/datasets/scjnugacj/scjn_dataset_ner).

## Benchmarking

```bash
        label        prec        rec          f1
Macro-average   0.9361195  0.9294152   0.9368145 
Micro-average   0.9856711  0.9857456   0.9851656  
```