---
layout: model
title: NER Model for 10 High Resourced Languages
author: John Snow Labs
name: xlm_roberta_large_token_classifier_hrl
date: 2021-12-26
tags: [arabic, german, english, spanish, french, italian, latvian, dutch, portuguese, chinese, xlm, roberta, ner, xx, open_source]
task: Named Entity Recognition
language: xx
edition: Spark NLP 3.3.4
spark_version: 2.4
supported: true
article_header:
  type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model has been fine-tuned for 10 high resourced languages (Arabic, German, English, Spanish, French, Italian, Latvian, Dutch, Portuguese and Chinese), leveraging `XLM-RoBERTa` embeddings and `XlmRobertaForTokenClassification` for NER purposes.

## Predicted Entities

`ORG`, `PER`, `LOC`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_HRL/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_HRL.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/xlm_roberta_large_token_classifier_hrl_xx_3.3.4_2.4_1640520352673.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
documentAssembler = DocumentAssembler()\
      .setInputCol("text")\
      .setOutputCol("document")

sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")\
       .setInputCols(["document"])\
       .setOutputCol("sentence")

tokenizer = Tokenizer()\
      .setInputCols(["sentence"])\
      .setOutputCol("token")

tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_large_token_classifier_hrl", "xx"))\
  .setInputCols(["sentence",'token'])\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(["sentence", "token", "ner"])\
      .setOutputCol("ner_chunk")
      
nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter])

empty_data = spark.createDataFrame([[""]]).toDF("text")

model = nlpPipeline.fit(empty_data)
text = """يمكنكم مشاهدة أمير منطقة الرياض الأمير فيصل بن بندر بن عبد العزيز في كل مناسبة وافتتاح تتعلق بمشاريع التعليم والصحة وخدمة الطرق والمشاريع الثقافية في منطقة الرياض."""
result = model.transform(spark.createDataFrame([[text]]).toDF("text"))
```
```scala
val documentAssembler = DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("document")

val sentenceDetector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx")
       .setInputCols(Array("document"))
       .setOutputCol("sentence")

val tokenizer = Tokenizer()
      .setInputCols(Array("sentence"))
      .setOutputCol("token")

val tokenClassifier = XlmRoBertaForTokenClassification.pretrained("xlm_roberta_large_token_classifier_hrl", "xx"))\
  .setInputCols(Array("sentence","token"))\
  .setOutputCol("ner")

ner_converter = NerConverter()\
      .setInputCols(Array("sentence", "token", "ner"))\
      .setOutputCol("ner_chunk")
      
val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, tokenClassifier, ner_converter))

val example = Seq.empty["يمكنكم مشاهدة أمير منطقة الرياض الأمير فيصل بن بندر بن عبد العزيز في كل مناسبة وافتتاح تتعلق بمشاريع التعليم والصحة وخدمة الطرق والمشاريع الثقافية في منطقة الرياض."].toDS.toDF("text")

val result = pipeline.fit(example).transform(example)
```
</div>

## Results

```bash
+---------------------------+---------+
|chunk                      |ner_label|
+---------------------------+---------+
|الرياض                     |LOC      |
|فيصل بن بندر بن عبد العزيز |PER      |
|الرياض                     |LOC      |
+---------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|xlm_roberta_large_token_classifier_hrl|
|Compatibility:|Spark NLP 3.3.4+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|xx|
|Size:|1.8 GB|
|Case sensitive:|true|
|Max sentense length:|256|

## Data Source

|**Language**|**Dataset**                |
|------------|---------------------------|
| Arabic     | [ANERcorp](https://camel.abudhabi.nyu.edu/anercorp/)                                                  |
| German     | [conll 2003](https://www.clips.uantwerpen.be/conll2003/ner/)                                          |
| English    | [conll 2003](https://www.clips.uantwerpen.be/conll2003/ner/)                                          |
| Spanish    | [conll 2002](https://www.clips.uantwerpen.be/conll2002/ner/)                                          |
| French     | [Europeana Newspapers](https://github.com/EuropeanaNewspapers/ner-corpora/tree/master/enp_FR.bnf.bio) |
| Italian    | [Italian I-CAB](https://ontotext.fbk.eu/icab.html)                                                    |
| Latvian    | [Latvian NER](https://github.com/LUMII-AILab/FullStack/tree/master/NamedEntities)                     |
| Dutch      | [conll 2002](https://www.clips.uantwerpen.be/conll2002/ner/)                                          |
| Portuguese | [Paramopama + Second Harem](https://github.com/davidsbatista/NER-datasets/tree/master/Portuguese)     |
| Chinese    | [MSRA](https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/MSRA)                              |
