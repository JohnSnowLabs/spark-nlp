---
layout: model
title: Legal Swiss Judgements Classification (French)
author: John Snow Labs
name: legclf_bert_swiss_judgements
date: 2022-10-27
tags: [fr, legal, licensed, sequence_classification]
task: Text Classification
language: fr
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Bert-based model that can be used to classify Swiss Judgement documents in French language into the following 6 classes according to their case area. It has been trained with SOTA approach.

## Predicted Entities

`public law`, `civil law`, `insurance law`, `social law`, `penal law`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_bert_swiss_judgements_fr_1.0.0_3.0_1666866243544.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_bert_swiss_judgements_fr_1.0.0_3.0_1666866243544.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
document_assembler = nlp.DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")

tokenizer = nlp.Tokenizer()\
    .setInputCols(["document"])\
    .setOutputCol("token")

clf_model = legal.BertForSequenceClassification.pretrained("legclf_bert_swiss_judgements", "fr", "legal/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("class")\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

clf_pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    clf_model   
])

data = spark.createDataFrame([["""Résumé : A. X. 1948) et Z. Ils se sont mariés à la xxxx 1992. Le mariage est resté sans enfants. T._ est, cependant, le père des enfants divorcés S._ et T._ (geb. 2004 et 2006). Après la suppression du budget commun, la vie séparée a dû être réglée. Disponible du 17. En décembre 2010, le président de la Cour de justice, Dorneck-Thierstein, a autorisé les époux à se séparer. Dans la mesure où cela est encore important, le juge a obligé le mari, pour l'année 2010 encore Fr. 3'000.-- à payer l'entretien de sa femme (Ziff. 3 ) De même, Z._ a été condamné, X._ à partir de janvier 2011 pour la durée ultérieure de la séparation une contribution de subsistance mensuelle de Fr. 7'085.-- de vous dépenser et de vous payer, en outre, la moitié du bonus net versé à chacun immédiatement après sa destination (Ziff. 4 ) En outre, le président de la Cour a ordonné la séparation des marchandises (Ziff. 5), dispose de la compétition du parti ou Les frais d’avocat (Ziff. 9) et impose les frais judiciaires à la moitié des deux parties (Ziff. 10 ) B. À l’encontre de cette décision, X._ a fait appel à la Cour suprême du canton de Solothurn. Elle a demandé de supprimer les paragraphes 3, 4, 5, 9 et 10 de la décision de première instance, et a présenté les demandes juridiques suivantes: Le mari est tenu de l'engager pour la période à partir de 21. Septembre 2009 à la fin du mois de décembre 2010 une contribution supplémentaire de Fr. 34'400.-- pour rembourser; pour la vie séparée à partir de janvier 2011, elle est dotée d'une contribution de subsistance de Fr. 10'000.-- pour recevoir par mois. La distribution des marchandises est de 21. Déposer en septembre 2010. En conclusion, le conjoint doit payer une contribution de parti raisonnable d'au moins Fr. 6'000.-- et pour payer tous les frais de justice. La Cour suprême du canton de Solothurn a déposé le recours à l'arrêt du 18. en mai 2011. C. À ce titre, X._ (ci-après dénommée « plaignante ») procède à la Cour fédérale. Dans sa plainte du 20. En juin 2011, elle présente la demande, la décision de la Cour suprême du canton Solothurn du 18. annuler en mai 2011 et répéter les demandes légales qu’elle a présentées devant la Cour suprême (cf. Bst. B ) En outre, il demande que la séparation des marchandises soit plus égalitaire par 7. Décembre 2010 à ordonner. Aucune consultation n’a été faite, mais les actes préjudiciels ont été reçus."""]]).toDF("text")

result = clf_pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+---------+
|                                                                                            document|    class|
+----------------------------------------------------------------------------------------------------+---------+
|Résumé : A. X. 1948) et Z. Ils se sont mariés à la xxxx 1992. Le mariage est resté sans enfants. ...|civil law|
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
|Language:|fr|
|Size:|390.0 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Training data is available [here](https://zenodo.org/record/7109926#.Y1gJwexBw8E).

## Benchmarking

```bash
| label         | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| civil-law     | 0.81      | 0.97   | 0.88     | 869     |
| insurance-law | 0.95      | 0.94   | 0.95     | 790     |
| other         | 1.00      | 0.40   | 0.57     | 15      |
| penal-law     | 0.94      | 0.91   | 0.93     | 1077    |
| public-law    | 0.93      | 0.85   | 0.89     | 1259    |
| social-law    | 0.94      | 0.95   | 0.95     | 834     |
| accuracy      |   -       |   -    | 0.91     | 4844    |
| macro-avg     | 0.93      | 0.84   | 0.86     | 4844    |
| weighted-avg  | 0.92      | 0.91   | 0.91     | 4844    |
```
