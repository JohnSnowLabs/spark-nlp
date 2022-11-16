---
layout: model
title: Detect Entities (66-labeled) in General Scope (Few-NERD dataset)
author: John Snow Labs
name: nerdl_fewnerd_subentity_100d
date: 2021-07-22
tags: [ner, en, fewnerd, public, open_source]
task: Named Entity Recognition
language: en
edition: Spark NLP 3.1.1
spark_version: 2.4
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

This model is trained on Few-NERD/inter public dataset and it extracts 66 entities that are in general scope.

## Predicted Entities

`building-theater`, `art-other`, `location-bodiesofwater`, `other-god`, `organization-politicalparty`, `product-other`, `building-sportsfacility`, `building-restaurant`, `organization-sportsleague`, `event-election`, `organization-media/newspaper`, `product-software`, `other-educationaldegree`, `person-politician`, `person-soldier`, `other-disease`, `product-airplane`, `person-athlete`, `location-mountain`, `organization-company`, `other-biologything`, `location-other`, `other-livingthing`, `person-actor`, `organization-other`, `event-protest`, `art-film`, `other-award`, `other-astronomything`, `building-airport`, `product-food`, `person-other`, `event-disaster`, `product-weapon`, `event-sportsevent`, `location-park`, `product-ship`, `building-library`, `art-painting`, `building-other`, `other-currency`, `organization-education`, `person-scholar`, `organization-showorganization`, `person-artist/author`, `product-train`, `location-GPE`, `product-car`, `art-writtenart`, `event-attack/battle/war/militaryconflict`, `other-law`, `other-medical`, `organization-sportsteam`, `art-broadcastprogram`, `art-music`, `organization-government/governmentagency`, `other-language`, `event-other`, `person-director`, `other-chemicalthing`, `product-game`, `organization-religion`, `location-road/railway/highway/transit`, `location-island`, `building-hotel`, `building-hospital`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_FEW_NERD/){:.button.button-orange}
[Open in Colab](https://colab.research.google.com/github/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER_FewNERD.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerdl_fewnerd_subentity_100d_en_3.1.1_2.4_1626970707030.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...

embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")\
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

ner = NerDLModel.pretrained("nerdl_fewnerd_subentity_100d") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(['document', 'token', 'ner']) \
.setOutputCol('ner_chunk')

nlp_pipeline = Pipeline(stages=[document_assembler, sentencer, tokenizer, embeddings, ner, ner_converter])

l_model = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = l_model.fullAnnotate("""12 Corazones ('12 Hearts') is Spanish-language dating game show produced in the United States for the television network Telemundo since January 2005, based on its namesake Argentine TV show format. The show is filmed in Los Angeles and revolves around the twelve Zodiac signs that identify each contestant. In 2008, Ho filmed a cameo in the Steven Spielberg feature film The Cloverfield Paradox, as a news pundit.""")
```
```scala
...

val embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")
.setInputCols(Array("sentence", "token"))
.setOutputCol("embeddings")

val ner = NerDLModel.pretrained("nerdl_fewnerd_subentity_100d")
.setInputCols(Array("sentence", "token", "embeddings")).setOutputCol("ner")

val ner_converter = NerConverter.setInputCols(Array("document", "token", "ner")) 
.setOutputCol("ner_chunk")

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, ner, ner_converter))
val data = Seq("12 Corazones ('12 Hearts') is Spanish-language dating game show produced in the United States for the television network Telemundo since January 2005, based on its namesake Argentine TV show format. The show is filmed in Los Angeles and revolves around the twelve Zodiac signs that identify each contestant. In 2008, Ho filmed a cameo in the Steven Spielberg feature film The Cloverfield Paradox, as a news pundit.").toDF("text")

val result = pipeline.fit(data).transform(data)
```


{:.nlu-block}
```python
import nlu
nlu.load("en.ner.fewnerd_subentity").predict("""12 Corazones ('12 Hearts') is Spanish-language dating game show produced in the United States for the television network Telemundo since January 2005, based on its namesake Argentine TV show format. The show is filmed in Los Angeles and revolves around the twelve Zodiac signs that identify each contestant. In 2008, Ho filmed a cameo in the Steven Spielberg feature film The Cloverfield Paradox, as a news pundit.""")
```

</div>

## Results

```bash
+-----------------------+----------------------------+
|chunk                  |ner_label                   |
+-----------------------+----------------------------+
|Corazones ('12 Hearts')|art-broadcastprogram        |
|Spanish-language       |other-language              |
|United States          |location-GPE                |
|Telemundo              |organization-media/newspaper|
|Argentine TV           |organization-media/newspaper|
|Los Angeles            |location-GPE                |
|Steven Spielberg       |person-director             |
|Cloverfield Paradox    |art-film                    |
+-----------------------+----------------------------+
```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerdl_fewnerd_subentity_100d|
|Type:|ner|
|Compatibility:|Spark NLP 3.1.1+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[sentence, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

Few-NERD:A Few-shot Named Entity Recognition Dataset, author: Ding, Ning and Xu, Guangwei and Chen, Yulin, and Wang, Xiaobin and Han, Xu and Xie, Pengjun and Zheng, Hai-Tao and Liu, Zhiyuan, book title: ACL-IJCNL, 2021.

## Benchmarking

```bash
+--------------------+-------+------+-------+-------+---------+------+------+
|              entity|     tp|    fp|     fn|  total|precision|recall|    f1|
+--------------------+-------+------+-------+-------+---------+------+------+
|            disaster|  309.0| 114.0|  287.0|  596.0|   0.7305|0.5185|0.6065|
|                film| 1589.0| 725.0|  810.0| 2399.0|   0.6867|0.6624|0.6743|
|            mountain|  851.0| 175.0|  431.0| 1282.0|   0.8294|0.6638|0.7374|
|            currency|  280.0|  66.0|  189.0|  469.0|   0.8092| 0.597|0.6871|
|             scholar|   31.0|  12.0|  413.0|  444.0|   0.7209|0.0698|0.1273|
|              island|  829.0| 165.0|  372.0| 1201.0|    0.834|0.6903|0.7554|
|      politicalparty|  242.0|  86.0|  283.0|  525.0|   0.7378| 0.461|0.5674|
|                ship|  461.0| 207.0|  311.0|  772.0|   0.6901|0.5972|0.6403|
|               award| 2234.0| 279.0| 1245.0| 3479.0|    0.889|0.6421|0.7457|
|    showorganization|  120.0| 201.0|  273.0|  393.0|   0.3738|0.3053|0.3361|
|            religion|  218.0| 117.0|  415.0|  633.0|   0.6507|0.3444|0.4504|
|           education| 5788.0| 852.0| 1001.0| 6789.0|   0.8717|0.8526| 0.862|
|                park|  259.0| 295.0|  176.0|  435.0|   0.4675|0.5954|0.5238|
|            painting|    0.0|   0.0|   14.0|   14.0|      0.0|   0.0|   0.0|
|               hotel|  570.0| 150.0|  254.0|  824.0|   0.7917|0.6917|0.7383|
|             library|  218.0|  92.0|  134.0|  352.0|   0.7032|0.6193|0.6586|
|         livingthing|  576.0| 280.0|  312.0|  888.0|   0.6729|0.6486|0.6606|
|   educationaldegree|  189.0|  31.0|   47.0|  236.0|   0.8591|0.8008|0.8289|
|            director|  673.0| 227.0|  507.0| 1180.0|   0.7478|0.5703|0.6471|
|                food|  474.0| 375.0|  341.0|  815.0|   0.5583|0.5816|0.5697|
|             athlete| 1181.0| 529.0|  540.0| 1721.0|   0.6906|0.6862|0.6884|
|            software|  922.0| 460.0|  493.0| 1415.0|   0.6671|0.6516|0.6593|
|             protest|  162.0| 212.0|  275.0|  437.0|   0.4332|0.3707|0.3995|
|               other|12555.0|7510.0|14369.0|26924.0|   0.6257|0.4663|0.5344|
|        sportsleague| 1439.0| 654.0|  842.0| 2281.0|   0.6875|0.6309| 0.658|
|            airplane| 1295.0| 442.0|  463.0| 1758.0|   0.7455|0.7366|0.7411|
|               train|  135.0| 111.0|  198.0|  333.0|   0.5488|0.4054|0.4663|
|        biologything| 1574.0| 625.0|  924.0| 2498.0|   0.7158|0.6301|0.6702|
|          politician| 3107.0|1545.0| 1688.0| 4795.0|   0.6679| 0.648|0.6578|
|               music|  419.0| 211.0|  182.0|  601.0|   0.6651|0.6972|0.6807|
|government/govern...|  564.0| 656.0|  511.0| 1075.0|   0.4623|0.5247|0.4915|
|     media/newspaper| 1600.0|1072.0|  893.0| 2493.0|   0.5988|0.6418|0.6196|
|               actor|  674.0| 161.0|  274.0|  948.0|   0.8072| 0.711| 0.756|
|            language|  698.0| 226.0|  335.0| 1033.0|   0.7554|0.6757|0.7133|
|       chemicalthing|  592.0| 231.0|  687.0| 1279.0|   0.7193|0.4629|0.5633|
|      sportsfacility|  870.0| 334.0|  291.0| 1161.0|   0.7226|0.7494|0.7357|
|            hospital|  226.0| 472.0|   49.0|  275.0|   0.3238|0.8218|0.4645|
|          writtenart|  297.0| 203.0|  450.0|  747.0|    0.594|0.3976|0.4763|
|road/railway/high...| 3238.0| 926.0| 1063.0| 4301.0|   0.7776|0.7528| 0.765|
|            election|   13.0|  13.0|  127.0|  140.0|      0.5|0.0929|0.1566|
|             soldier|  623.0| 537.0|  559.0| 1182.0|   0.5371|0.5271| 0.532|
|                 god|  332.0| 157.0|  414.0|  746.0|   0.6789| 0.445|0.5377|
|      astronomything| 1120.0| 353.0|  232.0| 1352.0|   0.7604|0.8284|0.7929|
|attack/battle/war...| 2516.0| 444.0|  590.0| 3106.0|     0.85|  0.81|0.8295|
|    broadcastprogram| 1056.0| 762.0|  811.0| 1867.0|   0.5809|0.5656|0.5731|
|             airport|  857.0|  96.0|  112.0|  969.0|   0.8993|0.8844|0.8918|
|             theater|   72.0|  31.0|  119.0|  191.0|    0.699| 0.377|0.4898|
|              weapon|  303.0| 190.0|  237.0|  540.0|   0.6146|0.5611|0.5866|
|             company| 5849.0|2632.0| 2570.0| 8419.0|   0.6897|0.6947|0.6922|
|                 car|  413.0| 293.0|  207.0|  620.0|    0.585|0.6661|0.6229|
|       artist/author| 4172.0|1953.0| 1777.0| 5949.0|   0.6811|0.7013|0.6911|
|             medical|   94.0| 112.0|  192.0|  286.0|   0.4563|0.3287|0.3821|
|             disease| 1009.0| 476.0|  447.0| 1456.0|   0.6795| 0.693|0.6862|
|                game|  141.0| 120.0|  264.0|  405.0|   0.5402|0.3481|0.4234|
|         sportsevent| 1042.0| 553.0|  552.0| 1594.0|   0.6533|0.6537|0.6535|
|          sportsteam| 3657.0|1133.0| 1301.0| 4958.0|   0.7635|0.7376|0.7503|
|          restaurant|  285.0| 444.0|  201.0|  486.0|   0.3909|0.5864|0.4691|
|       bodiesofwater|  314.0|  91.0|  343.0|  657.0|   0.7753|0.4779|0.5913|
|                 law| 1626.0| 583.0|  329.0| 1955.0|   0.7361|0.8317| 0.781|
|                 GPE|22173.0|5585.0| 3839.0|26012.0|   0.7988|0.8524|0.8247|
+--------------------+-------+------+-------+-------+---------+------+------+

+-----------------+
|            macro|
+-----------------+
|0.608599546406531|
+-----------------+

+-----------------+
|            micro|
+-----------------+
|0.684720504256685|
+-----------------+
```