---
layout: model
title: NER Model for Hindi+English
author: John Snow Labs
name: bert_token_classifier_hi_en_ner
date: 2021-12-27
tags: [ner, hi, en, open_source]
task: Named Entity Recognition
language: hi
edition: Spark NLP 3.2.0
spark_version: 3.0
supported: true
annotator: BertForTokenClassification
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model was imported from Hugging Face to carry out Name Entity Recognition with mixed Hindi-English texts, provided by the LinCE repository.

## Predicted Entities



{:.btn-box}
<button class="button button-orange" disabled>Live Demo</button>
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_HINDI_ENGLISH.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/bert_token_classifier_hi_en_ner_hi_3.2.0_3.0_1640612846736.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/bert_token_classifier_hi_en_ner_hi_3.2.0_3.0_1640612846736.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
document_assembler = DocumentAssembler()\
.setInputCol('text')\
.setOutputCol('document')

sentence_detector = SentenceDetector() \
.setInputCols(['document'])\
.setOutputCol('sentence')

tokenizer = Tokenizer()\
.setInputCols(['sentence']) \
.setOutputCol('token')

tokenClassifier_loaded = BertForTokenClassification.pretrained("bert_token_classifier_hi_en_ner","hi")\
.setInputCols(["sentence",'token'])\
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(["sentence","token","ner"])\
.setOutputCol("ner_chunk")

nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, tokenClassifier_loaded, ner_converter])

pipeline_model = nlp_pipeline.fit(spark.createDataFrame([[""]]).toDF("text"))

result = pipeline_model.transform(spark.createDataFrame([["""रिलायंस इंडस्ट्रीज़ लिमिटेड (Reliance Industries Limited) एक भारतीय संगुटिका नियंत्रक कंपनी है, जिसका मुख्यालय मुंबई, महाराष्ट्र (Maharashtra) में स्थित है।रतन नवल टाटा (28 दिसंबर 1937, को मुम्बई (Mumbai), में जन्मे) टाटा समुह के वर्तमान अध्यक्ष, जो भारत की सबसे बड़ी व्यापारिक समूह है, जिसकी स्थापना जमशेदजी टाटा ने की और उनके परिवार की पीढियों ने इसका विस्तार किया और इसे दृढ़ बनाया।"""]], ["text"]))
```
```scala
val document_assembler = DocumentAssembler() 
.setInputCol("text") 
.setOutputCol("document")

val sentence_detector  = SentenceDetector()
.setInputCols("document")
.setOutputCol("sentence")

val tokenizer = Tokenizer()
.setInputCols("sentence") 
.setOutputCol("token")

val tokenClassifier_loaded  = BertForTokenClassification.pretrained("bert_token_classifier_hi_en_ner","hi")
.setInputCols("sentence","token")
.setOutputCol("ner")

val ner_converter = NerConverter()
.setInputCols("sentence","token","ner")
.setOutputCol("ner_chunk")

val nlp_pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, tokenClassifier_loaded, ner_converter))

val data = Seq("""रिलायंस इंडस्ट्रीज़ लिमिटेड (Reliance Industries Limited) एक भारतीय संगुटिका नियंत्रक कंपनी है, जिसका मुख्यालय मुंबई, महाराष्ट्र (Maharashtra) में स्थित है।रतन नवल टाटा (28 दिसंबर 1937, को मुम्बई (Mumbai), में जन्मे) टाटा समुह के वर्तमान अध्यक्ष, जो भारत की सबसे बड़ी व्यापारिक समूह है, जिसकी स्थापना जमशेदजी टाटा ने की और उनके परिवार की पीढियों ने इसका विस्तार किया और इसे दृढ़ बनाया।""").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("hi.ner.bert").predict("""रिलायंस इंडस्ट्रीज़ लिमिटेड (Reliance Industries Limited) एक भारतीय संगुटिका नियंत्रक कंपनी है, जिसका मुख्यालय मुंबई, महाराष्ट्र (Maharashtra) में स्थित है।रतन नवल टाटा (28 दिसंबर 1937, को मुम्बई (Mumbai), में जन्मे) टाटा समुह के वर्तमान अध्यक्ष, जो भारत की सबसे बड़ी व्यापारिक समूह है, जिसकी स्थापना जमशेदजी टाटा ने की और उनके परिवार की पीढियों ने इसका विस्तार किया और इसे दृढ़ बनाया।""")
```

</div>

## Results

```bash
| chunk                 	| ner_label    	|
|-----------------------	|--------------	|
| रिलायंस इंडस्ट्रीज़ लिमिटेड           | ORGANISATION  |
| Reliance Industries Limited   | ORGANISATION  |
| मुंबई                   	| PLACE        	|
| महाराष्ट्र              	        | PLACE        	|
| Maharashtra                   | PLACE         | 
| नवल टाटा              	| PERSON       	|
| मुम्बई                  	| PLACE        	|
| Mumbai                        | PLACE         |
| टाटा समुह              	| ORGANISATION 	|
| भारत                  	| PLACE        	|
| जमशेदजी टाटा           	| PERSON       	|
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|bert_token_classifier_hi_en_ner|
|Compatibility:|Spark NLP 3.2.0+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token]|
|Output Labels:|[ner]|
|Language:|hi|
|Size:|665.7 MB|
|Case sensitive:|true|
|Max sentense length:|128|

## Data Source

https://ritual.uh.edu/lince/home