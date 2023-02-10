---
layout: model
title: Legal Swiss Judgements Classification (English)
author: John Snow Labs
name: legclf_bert_swiss_judgements
date: 2022-10-25
tags: [en, legal, licensed, sequence_classification]
task: Text Classification
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is a Bert-based model that can be used to classify Swiss Judgement documents into the following 6 classes according to their case area. It has been trained with SOTA approach.

## Predicted Entities

`public law`, `civil law`, `insurance law`, `social law`, `penal law`, `other`

{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legclf_bert_swiss_judgements_en_1.0.0_3.0_1666723020261.zip){:.button.button-orange.button-orange-trans.arr.button-icon.hidden}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/legal/models/legclf_bert_swiss_judgements_en_1.0.0_3.0_1666723020261.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

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

clf_model = legal.BertForSequenceClassification.pretrained("legclf_bert_swiss_judgements", "en", "legal/models")\
    .setInputCols(['document','token'])\
    .setOutputCol('class')\
    .setCaseSensitive(True)\
    .setMaxSentenceLength(512)

clf_pipeline = Pipeline(stages=[
    document_assembler, 
    tokenizer,
    clf_model   
])

data = spark.createDataFrame([["""Facts of fact: A. The Canton Police arrested X._ on 2. January 2007 due to suspicion of having committed an intrusive bull. In the trial of the trial 3. In January 2007, he agreed to have, together with a complicient, carried out a rubbish steel in a Jeans store in the fountain. After that, the investigative judge opened to him orally, he took him into investigative detention for the risk of collusion and continuation. X._ renounced a written and justified order, but desired a review of the investigation by the president of the Canton Court. by 4. In January 2007, the investigative judge submitted the documents to the president of the Canton Court with the request to withdraw the complaint and maintain the investigative detention. X._ requested to withdraw the investigative detention and immediately release him into freedom. He may be released under conditions or conditions. At its disposal of 5. In January 2007, the president of the Canton Court stated that the urgent offence was suspected in relation to the authorized invasion of the Jeans business and other invasions already occurred during a previous imprisonment. The risk of collusion is not accepted, but the recurrence forecast is extremely disadvantaged, therefore there is a risk of continuation. This is the request of the investigative judge - this is according to the instructions of 23. May 2006 (GG 2006 2; www.kgsz.ch) was not authorized to order investigative detention - to carry out and to confirm the investigative detention. At its disposal of 5. In January 2007, the president of the Canton Court stated that the urgent offence was suspected in relation to the authorized invasion of the Jeans business and other invasions already occurred during a previous imprisonment. The risk of collusion is not accepted, but the recurrence forecast is extremely disadvantaged, therefore there is a risk of continuation. This is the request of the investigative judge - this is according to the instructions of 23. May 2006 (GG 2006 2; www.kgsz.ch) was not authorized to order investigative detention - to carry out and to confirm the investigative detention. B. With complaint in criminal cases of 5. February 2007 requested X._: 1. It should be noted that the order GP 2007 3 of the Canton Court President of the Canton of Schwyz of 5. January 2007 is invalid and the complainant must be immediately released from prison. 2nd Eventually the order GP 2007 3 of the Canton Court President of the Canton of Schwyz of 5. January 2007 shall be repealed and the complainant shall be immediately released from investigative detention. and 3. Subeventual is the complainant due to the violation of the cantonal Swiss law by the instructions of the Canton Court of Schwyz of 23. May 2006 immediately released from the detention. Fourth All under cost and compensation consequences at the expense of the complainant.” Fourth All under cost and compensation consequences at the expense of the complainant.” C. The investigative judge requires in his judgment that “there must be established that the investigative detention was ordered by the investigative authority in accordance with the law and that the appeal submitted by the Court of Appeal with the approval of the request for responsibility and the confirmation of the investigative detention (Decree of the President of the Canton Court of 5 January 2007) has been legally rejected.” Insofar as X._ requires his immediate release, the complaint must be rejected. The President of the Canton Court asks to reject the complaint insofar as it is necessary. X._ requires unpaid legal assistance and defence and completes in its response to the complaint."""]]).toDF("text")

result = clf_pipeline.fit(data).transform(data)
```

</div>

## Results

```bash
+----------------------------------------------------------------------------------------------------+----------+
|                                                                                            document|     class|
+----------------------------------------------------------------------------------------------------+----------+
|Facts of fact: A. The Canton Police arrested X._ on 2. January 2007 due to suspicion of having co...|public law|
+----------------------------------------------------------------------------------------------------+----------+
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
|Language:|en|
|Size:|409.7 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Training data is available [here](https://zenodo.org/record/7109926#.Y1gJwexBw8E).

## Benchmarking

```bash
| label         | precision | recall | f1-score | support |
|---------------|-----------|--------|----------|---------|
| civil-law     | 0.97      | 0.96   | 0.96     | 1189    |
| insurance-law | 0.95      | 0.98   | 0.96     | 1081    |
| other         | 0.92      | 0.90   | 0.91     | 40      |
| penal-law     | 0.97      | 0.94   | 0.96     | 1140    |
| public-law    | 0.94      | 0.97   | 0.95     | 1551    |
| social-law    | 0.98      | 0.94   | 0.96     | 970     |
| accuracy      |   -       |   -    | 0.96     | 5971    |
| macro-avg     | 0.95      | 0.95   | 0.95     | 5971    |
| weighted-avg  | 0.96      | 0.96   | 0.96     | 5971    |
```
