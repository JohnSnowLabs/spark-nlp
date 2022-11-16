---
layout: model
title: Legal Zero-shot NER
author: John Snow Labs
name: legner_roberta_zeroshot
date: 2022-09-02
tags: [en, legal, ner, zero, shot, zeroshot, licensed]
task: Named Entity Recognition
language: en
edition: Legal NLP 1.0.0
spark_version: 3.0
supported: true
recommended: true
annotator: ZeroShotNER
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained to carry out a Zero-Shot Named Entity Recognition (NER) approach, detecting any kind of entities with no training dataset, just tje pretrained RoBERTa embeddings (included in the model) and some examples.

## Predicted Entities



{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/legal/LEGNER_ZEROSHOT/){:.button.button-orange}
<button class="button button-orange" disabled>Open in Colab</button>
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/legal/models/legner_roberta_zeroshot_en_1.0.0_3.2_1662113815288.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
documentAssembler = nlp.DocumentAssembler()\
  .setInputCol("text")\
  .setOutputCol("document")

sparktokenizer = nlp.Tokenizer()\
  .setInputCols("document")\
  .setOutputCol("token")

zero_shot_ner = legal.ZeroShotNerModel.pretrained("legner_roberta_zeroshot", "en", "legal/models")\
    .setInputCols(["document", "token"])\
    .setOutputCol("zero_shot_ner")\
    .setEntityDefinitions(
        {
            "DATE": ['When was the company acquisition?', 'When was the company purchase agreement?', "When was the agreement?"],
            "ORG": ["Which company?"],
            "STATE": ["Which state?"],
            "AGREEMENT": ["What kind of agreement?"],
            "LICENSE": ["What kind of license?"],
            "LICENSE_RECIPIENT": ["To whom the license is granted?"]
        })

nerconverter = nlp.NerConverter()\
  .setInputCols(["document", "token", "zero_shot_ner"])\
  .setOutputCol("ner_chunk")

pipeline =  Pipeline(stages=[
  documentAssembler,
  sparktokenizer,
  zero_shot_ner,
  nerconverter,
    ]
)

sample_text = ["In March 2012, as part of a longer-term strategy, the Company acquired Vertro, Inc., which owned and operated the ALOT product portfolio.",
              "In February 2017, the Company entered into an asset purchase agreement with NetSeer, Inc.",
              "This INTELLECTUAL PROPERTY AGREEMENT, dated as of December 31, 2018 (the 'Effective Date') is entered into by and between Armstrong Flooring, Inc., a Delaware corporation ('Seller') and AFI Licensing LLC, a Delaware company('Licensing')"
              "The Company hereby grants to Seller a perpetual, non- exclusive, royalty-free license"]

p_model = pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

res = p_model.transform(spark.createDataFrame(sample_text, StringType()).toDF("text"))

res.select(F.explode(F.arrays_zip(res.ner_chunk.result, res.ner_chunk.begin, res.ner_chunk.end, res.ner_chunk.metadata)).alias("cols")) \
   .select(F.expr("cols['0']").alias("chunk"),
           F.expr("cols['3']['entity']").alias("ner_label"))\
   .filter("ner_label!='O'")\
   .show(truncate=False)
```

</div>

## Results

```bash
+---------------------------------------+-----------------+
|chunk                                  |ner_label        |
+---------------------------------------+-----------------+
|March 2012                             |DATE             |
|Vertro, Inc                            |ORG              |
|February 2017                          |DATE             |
|asset purchase agreement               |AGREEMENT        |
|NetSeer                                |ORG              |
|INTELLECTUAL PROPERTY AGREEMENT        |AGREEMENT        |
|December 31, 2018                      |DATE             |
|Armstrong Flooring                     |ORG              |
|Delaware                               |STATE            |
|AFI Licensing LLC                      |ORG              |
|Delaware                               |ORG              |
|Seller                                 |LICENSE_RECIPIENT|
|perpetual, non- exclusive, royalty-free|LICENSE          |
+---------------------------------------+-----------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|legner_roberta_zeroshot|
|Type:|legal|
|Compatibility:|Legal NLP 1.0.0+|
|License:|Licensed|
|Edition:|Official|
|Input Labels:|[document_question, document_context]|
|Output Labels:|[answer]|
|Language:|en|
|Size:|460.2 MB|
|Case sensitive:|true|
|Max sentence length:|512|

## References

Legal Roberta Embeddings
