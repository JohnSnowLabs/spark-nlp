---
layout: model
title: Legal Swiss Judgements Classification (Italian)
author: John Snow Labs
name: legclf_bert_swiss_judgements
date: 2022-10-25
tags: [it, legal, licensed, sequence_classification]
task: Text Classification
language: it
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Bert-based model that can be used to classify Swiss Judgement documents in Italian language into the following 6 classes according to their case area. It has been trained with SOTA approach.

## Predicted Entities

`public law`, `civil law`, `insurance law`, `social law`, `penal law`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_bert_swiss_judgements_it_1.0.0_3.0_1666710362375.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_bert_swiss_judgements_it_1.0.0_3.0_1666710362375.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler() \
    .setInputCol('text') \
    .setOutputCol('document')

tokenizer = nlp.Tokenizer()\
    .setInputCols(['document'])\
    .setOutputCol("token")

clf_model = legal.BertForSequenceClassification.pretrained("legclf_bert_swiss_judgements", "it", "legal/models")\
    .setInputCols(['document','token'])\
    .setOutputCol("class")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

clf_pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    clf_model   
])

data = spark.createDataFrame([["""Attualità: A. Disponibile dal 21. Nell'ottobre del 2004, l'Allianza di assicurazioni svizzere (in prosieguo: Allianz) ha messo in atto il R._ (geb. 1965) per le conseguenze di un incidente del 23. Nel mese di marzo del 2001 le prestazioni sono ritornate al 31. Nel mese di marzo del 2004 si è presentato la decisione del 6. Nel luglio del 2005 è stato arrestato. A. A disposizione del 21. Nell'ottobre del 2004, l'Allianza di assicurazioni svizzere (in prosieguo: Allianz) ha messo in atto il R._ (geb. 1965) per le conseguenze di un incidente del 23. Nel mese di marzo del 2001 le prestazioni sono ritornate al 31. Nel mese di marzo del 2004 si è presentato la decisione del 6. Nel luglio del 2005 è stato arrestato. di B. Il 7. Nel novembre 2005 R._ ha presentato una denuncia contro la decisione di interrogatorio al Tribunale amministrativo del Cantone di Schwyz. Con la lettera del 9. Nel novembre del 2005, il vicepresidente del Tribunale amministrativo ha informato gli assicurati che la denuncia è stata presentata in ritardo secondo la legge cantonale massiccia, il motivo per cui non è possibile procedere, e gli ha dato l'opportunità di pronunciarsi. Con l’ingresso del 15. Nel novembre 2005 R._ ha presentato una richiesta di ripristino del termine di reclamo. Con la decisione del 6. Nel dicembre 2005 il Tribunale amministrativo non ha presentato la denuncia. di B. Il 7. Nel novembre 2005 R._ ha presentato una denuncia contro la decisione di interrogatorio al Tribunale amministrativo del Cantone di Schwyz. Con la lettera del 9. Nel novembre del 2005, il vicepresidente del Tribunale amministrativo ha informato gli assicurati che la denuncia è stata presentata in ritardo secondo la legge cantonale massiccia, il motivo per cui non è possibile procedere, e gli ha dato l'opportunità di pronunciarsi. Con l’ingresso del 15. Nel novembre 2005 R._ ha presentato una richiesta di ripristino del termine di reclamo. Con la decisione del 6. Nel dicembre 2005 il Tribunale amministrativo non ha presentato la denuncia. C. Con un ricorso al Tribunale amministrativo, R._ chiede alla causa principale che, annullando la decisione pregiudiziale, il tribunale cantonale sia obbligato a presentare il ricorso del 7. di entrare nel novembre 2005. Dal punto di vista procedurale, il giudice può presentare la richiesta giuridica di aderire agli atti pregiudiziali e di ordinare un secondo cambio di scrittura. Il Tribunale amministrativo del Cantone di Schwyz e l'Alleanza concludono il ricorso alla Corte amministrativa. L’Ufficio federale per la salute rinuncia ad una consultazione."""]]).toDF("text")

result = clf_pipeline.fit(data).transform(data)

```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+-------------+
|                                                                                            document|        class|
+----------------------------------------------------------------------------------------------------+-------------+
|Attualità: A. Disponibile dal 21. Nell'ottobre del 2004, l'Allianza di assicurazioni svizzere (in...|insurance law|
+----------------------------------------------------------------------------------------------------+-------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legclf_bert_swiss_judgements|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document, token]|
|Output Labels:|[class]|
|Language:|it|
|Size:|396.5 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Training data is available [here](https://zenodo.org/record/7109926#.Y1gJwexBw8E).

## Benchmarking

```bash
| label         | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| civil-law     | 0.90      | 0.93   | 0.92     | 1144    |
| insurance-law | 0.93      | 0.96   | 0.95     | 1034    |
| other         | 0.88      | 0.44   | 0.58     | 32      |
| penal-law     | 0.91      | 0.95   | 0.93     | 1219    |
| public-law    | 0.93      | 0.88   | 0.90     | 1433    |
| social-law    | 0.96      | 0.92   | 0.94     | 924     |
| accuracy      |   -       |   -    | 0.92     | 5786    |
| macro-avg     | 0.92      | 0.85   | 0.87     | 5786    |
| weighted-avg  | 0.92      | 0.92   | 0.92     | 5786    |
```
