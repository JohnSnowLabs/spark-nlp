---
layout: model
title: Detect Persons, Locations and Organization Entities in Turkish (bert_multi_cased)
author: John Snow Labs
name: turkish_ner_bert
date: 2020-11-10
task: Named Entity Recognition
language: tr
edition: Spark NLP 2.6.2
spark_version: 2.4
tags: [tr, open_source]
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

{:.h2_title}
## Description

Pretrained Named Entity Recognition (NER) deep learning model for Turkish texts. It recognizes Persons, Locations, and Organization entities using multi-lingual Bert word embedding. The SparkNLP deep learning model (NerDL) is inspired by a former state of the art model for NER ç Chiu & Nicols, Named Entity Recognition with Bidirectional LSTM-CNN.

{:.h2_title}
## Predicted Entities
Persons-``PER``, Locations-``LOC``, Organizations-``ORG``.

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_TR/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_TR.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/turkish_ner_bert_tr_2.6.2_2.4_1605043368882.zip){:.button.button-orange.button-orange-trans.arr.button-icon}
[Copy S3 URI](s3://auxdata.johnsnowlabs.com/public/models/turkish_ner_bert_tr_2.6.2_2.4_1605043368882.zip){:.button.button-orange.button-orange-trans.button-icon.button-copy-s3}

## How to use

Use as part of an NLP pipeline with the following stages: DocumentAssembler, SentenceDetector, Tokenizer, WordEmbeddingsModel, NerDLModel. Add the NerConverter to the end of the pipeline to convert entity tokens into full entity chunks.

<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}

```python
...
ner_model = NerDLModel.pretrained("turkish_ner_bert", "tr") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")
...        
nlp_pipeline = Pipeline(stages=[document_assembler, sentence_detector, tokenizer, embeddings, ner_model, ner_converter])
pipeline_model = nlp_pipeline.fit(spark.createDataFrame([['']]).toDF('text'))

result = pipeline_model.transform(spark.createDataFrame([["William Henry Gates III (28 Ekim 1955 doğumlu), Amerikalı bir iş adamı, yazılım geliştirici, yatırımcı ve hayırseverdir. En çok Microsoft şirketinin kurucu ortağı olarak bilinir. William Gates , Microsoft şirketindeki kariyeri boyunca başkan, icra kurulu başkanı, başkan ve yazılım mimarisi başkanı pozisyonlarında bulunmuş, aynı zamanda Mayıs 2014'e kadar en büyük bireysel hissedar olmuştur. O, 1970'lerin ve 1980'lerin mikrobilgisayar devriminin en tanınmış girişimcilerinden ve öncülerinden biridir. Seattle Washington'da doğup büyüyen William Gates, 1975'te New Mexico Albuquerque'de çocukluk arkadaşı Paul Allen ile Microsoft şirketini kurdu; dünyanın en büyük kişisel bilgisayar yazılım şirketi haline geldi. William Gates, Ocak 2000'de icra kurulu başkanı olarak istifa edene kadar şirketi başkan ve icra kurulu başkanı olarak yönetti ve daha sonra yazılım mimarisi başkanı oldu. 1990'ların sonlarında, William Gates rekabete aykırı olduğu düşünülen iş taktikleri nedeniyle eleştirilmişti. Bu görüş, çok sayıda mahkeme kararıyla onaylanmıştır. Haziran 2006'da William Gates, Microsoft şirketinde yarı zamanlı bir göreve ve 2000 yılında eşi Melinda Gates ile birlikte kurdukları özel hayır kurumu olan B&Melinda G. Vakfı'nda tam zamanlı çalışmaya geçeceğini duyurdu. Görevlerini kademeli olarak Ray Ozzie ve Craig Mundie'ye devretti. Şubat 2014'te Microsoft başkanlığından ayrıldı ve yeni atanan icra kurulu başkanı, Satya Nadella'yı desteklemek için teknoloji danışmanı olarak yeni bir göreve başladı."]], ["text"]))
```

```scala
...
val ner_model = NerDLModel.pretrained("turkish_ner_bert", "tr")
.setInputCols(Array("sentence", "token", "embeddings"))
.setOutputCol("ner")
...
val pipeline = new Pipeline().setStages(Array(document_assembler, sentence_detector, tokenizer, embeddings, ner_model, ner_converter))

val data = Seq("William Henry Gates III (28 Ekim 1955 doğumlu), Amerikalı bir iş adamı, yazılım geliştirici, yatırımcı ve hayırseverdir. En çok Microsoft şirketinin kurucu ortağı olarak bilinir. William Gates , Microsoft şirketindeki kariyeri boyunca başkan, icra kurulu başkanı, başkan ve yazılım mimarisi başkanı pozisyonlarında bulunmuş, aynı zamanda Mayıs 2014"e kadar en büyük bireysel hissedar olmuştur. O, 1970"lerin ve 1980"lerin mikrobilgisayar devriminin en tanınmış girişimcilerinden ve öncülerinden biridir. Seattle Washington"da doğup büyüyen William Gates, 1975"te New Mexico Albuquerque"de çocukluk arkadaşı Paul Allen ile Microsoft şirketini kurdu; dünyanın en büyük kişisel bilgisayar yazılım şirketi haline geldi. William Gates, Ocak 2000"de icra kurulu başkanı olarak istifa edene kadar şirketi başkan ve icra kurulu başkanı olarak yönetti ve daha sonra yazılım mimarisi başkanı oldu. 1990"ların sonlarında, William Gates rekabete aykırı olduğu düşünülen iş taktikleri nedeniyle eleştirilmişti. Bu görüş, çok sayıda mahkeme kararıyla onaylanmıştır. Haziran 2006"da William Gates, Microsoft şirketinde yarı zamanlı bir göreve ve 2000 yılında eşi Melinda Gates ile birlikte kurdukları özel hayır kurumu olan B&Melinda G. Vakfı"nda tam zamanlı çalışmaya geçeceğini duyurdu. Görevlerini kademeli olarak Ray Ozzie ve Craig Mundie"ye devretti. Şubat 2014"te Microsoft başkanlığından ayrıldı ve yeni atanan icra kurulu başkanı, Satya Nadella'yı desteklemek için teknoloji danışmanı olarak yeni bir göreve başladı.").toDF("text")
val result = pipeline.fit(data).transform(data)
```

{:.nlu-block}
```python
import nlu
text = ["""William Henry Gates III (28 Ekim 1955 doğumlu), Amerikalı bir iş adamı, yazılım geliştirici, yatırımcı ve hayırseverdir. En çok Microsoft şirketinin kurucu ortağı olarak bilinir. William Gates , Microsoft şirketindeki kariyeri boyunca başkan, icra kurulu başkanı, başkan ve yazılım mimarisi başkanı pozisyonlarında bulunmuş, aynı zamanda Mayıs 2014'e kadar en büyük bireysel hissedar olmuştur. O, 1970'lerin ve 1980'lerin mikrobilgisayar devriminin en tanınmış girişimcilerinden ve öncülerinden biridir. Seattle Washington'da doğup büyüyen William Gates, 1975'te New Mexico Albuquerque'de çocukluk arkadaşı Paul Allen ile Microsoft şirketini kurdu; dünyanın en büyük kişisel bilgisayar yazılım şirketi haline geldi. William Gates, Ocak 2000'de icra kurulu başkanı olarak istifa edene kadar şirketi başkan ve icra kurulu başkanı olarak yönetti ve daha sonra yazılım mimarisi başkanı oldu. 1990'ların sonlarında, William Gates rekabete aykırı olduğu düşünülen iş taktikleri nedeniyle eleştirilmişti. Bu görüş, çok sayıda mahkeme kararıyla onaylanmıştır. Haziran 2006'da William Gates, Microsoft şirketinde yarı zamanlı bir göreve ve 2000 yılında eşi Melinda Gates ile birlikte kurdukları özel hayır kurumu olan B&Melinda G. Vakfı'nda tam zamanlı çalışmaya geçeceğini duyurdu. Görevlerini kademeli olarak Ray Ozzie ve Craig Mundie'ye devretti. Şubat 2014'te Microsoft başkanlığından ayrıldı ve yeni atanan icra kurulu başkanı, Satya Nadella'yı desteklemek için teknoloji danışmanı olarak yeni bir göreve başladı."""]

ner_df = nlu.load('tr.ner.bert').predict(text, output_level = "chunk")
ner_df[["entities", "entities_confidence"]]
```

</div>

{:.h2_title}
## Results

```bash
+-------------------------+---------+
|chunk                    |ner_label|
+-------------------------+---------+
|William Henry Gates III  |PER      |
|Microsoft                |ORG      |
|William Gates            |PER      |
|Microsoft                |ORG      |
|Seattle Washington'da    |LOC      |
|William Gates            |PER      |
|New Mexico Albuquerque'de|LOC      |
|Paul Allen               |PER      |
|Microsoft                |ORG      |
|William Gates            |PER      |
|William Gates            |PER      |
|William Gates            |PER      |
|Microsoft                |ORG      |
|Melinda Gates            |PER      |
|B&Melinda G. Vakfı'nda   |ORG      |
|Ray Ozzie                |PER      |
|Craig Mundie'ye          |PER      |
|Microsoft                |ORG      |
|Satya Nadella'yı         |PER      |
+-------------------------+---------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|turkish_ner_bert|
|Type:|ner|
|Compatibility:|Spark NLP 2.6.2+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|tr|
|Dependencies:|bert_multi_cased|

{:.h2_title}
## Data Source

Trained on a custom dataset with multi-lingual Bert Embeddings ``bert_multi_cased``.

{:.h2_title}
## Benchmarking

```bash
label	 tp	 fp	 fn	 prec	         rec	         f1
B-LOC	 1949	 156	 158	 0.92589074	 0.9250119	 0.9254511
I-ORG	 1266	 266	 98	 0.8263708	 0.9281525	 0.8743094
I-LOC	 270	 54	 79	 0.8333333	 0.77363896	 0.8023774
I-PER	 1507	 89	 94	 0.94423556	 0.9412867	 0.94275886
B-ORG	 1805	 242	 119	 0.88177824	 0.9381497	 0.90909094
B-PER	 2841	 152	 267	 0.9492148	 0.91409266	 0.93132275
tp: 9638 fp: 959 fn: 815 labels: 6
Macro-average	 prec: 0.8934706, rec: 0.90338874, f1: 0.89840233
Micro-average	 prec: 0.9095027, rec: 0.92203194, f1: 0.91572446
```