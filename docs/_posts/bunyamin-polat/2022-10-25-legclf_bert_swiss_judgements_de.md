---
layout: model
title: Legal Swiss Judgements Classification (German)
author: John Snow Labs
name: legclf_bert_swiss_judgements
date: 2022-10-25
tags: [de, legal, licensed, sequence_classification]
task: Text Classification
language: de
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Bert-based model that can be used to classify Swiss Judgement documents in German language into the following 6 classes according to their case area. It has been trained with SOTA approach.

## Predicted Entities

`public law`, `civil law`, `insurance law`, `social law`, `penal law`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_bert_swiss_judgements_de_1.0.0_3.0_1666721425728.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_bert_swiss_judgements_de_1.0.0_3.0_1666721425728.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

clf_model = legal.BertForSequenceClassification.pretrained("legclf_bert_swiss_judgements", "de", "legal/models")\
    .setInputCols(['document','token'])\
    .setOutputCol("class")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

clf_pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    clf_model   
])

data = spark.createDataFrame([["""Sachverhalt: A. Mit Strafbefehl vom 30. Juli 2015 sprach die Staatsanwaltschaft Lenzburg-Aarau gegen X._ eine bedingte Geldstrafe von 150 Tagessätzen zu Fr. 150.-- (Probezeit vier Jahre) sowie eine Busse von Fr. 4'500.-- aus wegen Führens eines Motorfahrzeugs in angetrunkenem Zustand sowie wegen mehrfacher Anstiftung zu falschem Zeugnis. Die Staatsanwaltschaft legte X._ unter anderem zur Last, am 5. Juli 2013 nach Aussage von Zeugen sein Auto mit einem Blutalkoholgehalt von mindestens 2,12 Promille bestiegen und von Lenzburg an seinen Wohnort in Z._ gelenkt zu haben. Das nach Einsprache von X._ mit der Sache befasste Bezirksgericht Lenzburg sprach ihn vom Vorwurf der mehrfachen Anstiftung zu falschem Zeugnis frei und verurteilte ihn wegen Führens eines Motorfahrzeugs in angetrunkenem Zustand zu einer bedingten Geldstrafe von 105 Tagessätzen zu Fr. 210.-- (Probezeit zwei Jahre) und zu einer Busse von Fr. 4'400.-- (Urteil vom 15. August 2016). B. X._ erhob Berufung. Das Obergericht des Kantons Aargau wies das Rechtsmittel ab (Urteil vom 3. Juli 2017). C. Mit Beschwerde in Strafsachen beantragt X._, das angefochtene Urteil sei aufzuheben und er von Schuld und Strafe freizusprechen."""]]).toDF("text")

result = clf_pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+---------+
|                                                                                            document|    class|
+----------------------------------------------------------------------------------------------------+---------+
|Sachverhalt: A. Mit Strafbefehl vom 30. Juli 2015 sprach die Staatsanwaltschaft Lenzburg-Aarau ge...|penal law|
+----------------------------------------------------------------------------------------------------+---------+
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
|Language:|de|
|Size:|409.6 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Training data is available [here](https://zenodo.org/record/7109926#.Y1gJwexBw8E).

## Benchmarking

```bash
| label         | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| civil-law     | 0.93      | 0.96   | 0.94     | 809     |
| insurance-law | 0.92      | 0.94   | 0.93     | 357     |
| other         | 0.76      | 0.70   | 0.73     | 23      |
| penal-law     | 0.97      | 0.95   | 0.96     | 913     |
| public-law    | 0.94      | 0.94   | 0.94     | 1048    |
| social-law    | 0.97      | 0.95   | 0.96     | 719     |
| accuracy      |   -       |   -    | 0.95     | 3869    |
| macro-avg     | 0.92      | 0.91   | 0.91     | 3869    |
| weighted-avg  | 0.95      | 0.95   | 0.95     | 3869    |                
```
