---
layout: model
title: Detect actions in general commands related to music, restaurant, movies.
author: John Snow Labs
name: nerdl_snips_100d
date: 2021-02-15
task: Named Entity Recognition
language: en
edition: Spark NLP 2.7.3
spark_version: 2.4
tags: [open_source, ner, en]
supported: true
annotator: NerDLModel
article_header:
type: cover
use_language_switcher: "Python-Scala-Java"
---

## Description

Understand user commands and find relevant entities and actions and tag them to get a structured representation for automation.

## Predicted Entities

`playlist_owner`, `served_dish`, `track`, `poi`, `cuisine`, `spatial_relation`, `object_type`, `facility`, `album`, `country`, `geographic_poi`, `location_name`, `object_part_of_series_type`, `object_select`, `artist`, `rating_value`, `best_rating`, `sort`, `party_size_description`, `party_size_number`, `restaurant_name`, `object_location_type`, `playlist`, `service`, `city`, `O`, `genre`, `movie_name`, `current_location`, `rating_unit`, `restaurant_type`, `condition_temperature`, `condition_description`, `entity_name`, `movie_type`, `object_name`, `state`, `year`, `music_item`, `timeRange`

{:.btn-box}
[Live Demo](https://demo.johnsnowlabs.com/public/NER_CLS_SNIPS){:.button.button-orange}
[Open in Colab](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/streamlit_notebooks/NER.ipynb){:.button.button-orange.button-orange-trans.co.button-icon}
[Download](https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/models/nerdl_snips_100d_en_2.7.3_2.4_1613403676821.zip){:.button.button-orange.button-orange-trans.arr.button-icon}

## How to use



<div class="tabs-box" markdown="1">
{% include programmingLanguageSelectScalaPythonNLU.html %}
```python
...

embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")\
.setInputCols("sentence", "token") \
.setOutputCol("embeddings")

ner = NerDLModel.pretrained("nerdl_snips_100d") \
.setInputCols(["sentence", "token", "embeddings"]) \
.setOutputCol("ner")

ner_converter = NerConverter()\
.setInputCols(['document', 'token', 'ner']) \
.setOutputCol('ner_chunk')

nlp_pipeline = Pipeline(stages=[document_assembler, sentencer, tokenizer, embeddings, ner, ner_converter])

l_model = LightPipeline(nlp_pipeline.fit(spark.createDataFrame([['']]).toDF("text")))

annotations = l_model.fullAnnotate('book a spot for nona gray  myrtle and alison at a top-rated brasserie that is distant from wilson av on nov  the 4th  2030 that serves ouzeri')

...
```
```scala
...

val embeddings = WordEmbeddingsModel.pretrained("glove_100d", "en")
.setInputCols(Array("sentence", 'token'))
.setOutputCol("embeddings")

val ner = NerDLModel.pretrained('nerdl_snips_100d')
.setInputCols(Array('sentence', 'token', 'embeddings')).setOutputCol('ner')

val ner_converter = NerConverter.setInputCols(Array('document', 'token', 'ner')) \
.setOutputCol('ner_chunk')

val pipeline = new Pipeline().setStages(Array(document_assembler, tokenizer, embeddings, ner, ner_converter))
val data = Seq("book a spot for nona gray  myrtle and alison at a top-rated brasserie that is distant from wilson av on nov  the 4th  2030 that serves ouzeri").toDF("text")
val result = pipeline.fit(data).transform(data)

...
```


{:.nlu-block}
```python
import nlu
nlu.load("en.classify.snips").predict("""book a spot for nona gray  myrtle and alison at a top-rated brasserie that is distant from wilson av on nov  the 4th  2030 that serves ouzeri""")
```

</div>

## Results

```bash
+------------------------------+------------------------+
| ner_chunk                    | label             	 	|
+------------------------------+------------------------+
| nona gray myrtle and alison  | PARTY_SIZE_DESCRIPTION |
| top-rated					   | SORT					|
| brasserie					   | RESTAURANT_TYPE		|
| distant					   | SPATIAL_RELATION		|
| wilson Macro-average         | POI					|
| nov the 4th 2030			   | TIMERANGE				|
| ouzeri					   | CUISINE				|
+------------------------------+------------------------+

```

{:.model-param}
## Model Information

{:.table-model}
|---|---|
|Model Name:|nerdl_snips_100d|
|Type:|ner|
|Compatibility:|Spark NLP 2.7.3+|
|License:|Open Source|
|Edition:|Official|
|Input Labels:|[document, token, embeddings]|
|Output Labels:|[ner]|
|Language:|en|

## Data Source

This model is trained on the NLU Benchmark SNIPS dataset https://github.com/MiuLab/SlotGated-SLU

## Benchmarking

```bash
B-facility	 3	 0	 0	 1.0	 1.0	 1.0
B-poi	 7	 0	 1	 1.0	 0.875	 0.93333334
B-object_location_type	 22	 1	 0	 0.95652175	 1.0	 0.9777778
B-service	 24	 2	 0	 0.9230769	 1.0	 0.96000004
I-entity_name	 53	 2	 1	 0.96363634	 0.9814815	 0.9724771
B-genre	 5	 0	 0	 1.0	 1.0	 1.0
I-service	 5	 0	 0	 1.0	 1.0	 1.0
I-object_type	 66	 0	 0	 1.0	 1.0	 1.0
I-sort	 9	 0	 0	 1.0	 1.0	 1.0
I-city	 19	 1	 0	 0.95	 1.0	 0.9743589
B-music_item	 102	 2	 2	 0.9807692	 0.9807692	 0.9807692
I-movie_name	 100	 5	 21	 0.95238096	 0.8264463	 0.8849558
B-party_size_description	 10	 0	 0	 1.0	 1.0	 1.0
B-served_dish	 10	 3	 2	 0.7692308	 0.8333333	 0.8
B-object_type	 161	 8	 1	 0.9526627	 0.99382716	 0.9728097
B-playlist	 123	 6	 6	 0.95348835	 0.95348835	 0.95348835
B-restaurant_name	 14	 1	 1	 0.93333334	 0.93333334	 0.93333334
B-geographic_poi	 11	 0	 0	 1.0	 1.0	 1.0
B-condition_description	 28	 0	 0	 1.0	 1.0	 1.0
I-object_location_type	 16	 0	 0	 1.0	 1.0	 1.0
B-spatial_relation	 70	 3	 1	 0.9589041	 0.9859155	 0.9722222
I-party_size_description	 35	 0	 0	 1.0	 1.0	 1.0
I-poi	 10	 0	 1	 1.0	 0.90909094	 0.95238096
I-artist	 111	 4	 1	 0.9652174	 0.9910714	 0.9779735
B-condition_temperature	 23	 0	 0	 1.0	 1.0	 1.0
I-movie_type	 16	 0	 0	 1.0	 1.0	 1.0
I-object_part_of_series_type	 0	 0	 1	 0.0	 0.0	 0.0
B-city	 60	 1	 0	 0.9836066	 1.0	 0.9917355
I-location_name	 29	 0	 1	 1.0	 0.96666664	 0.9830508
B-album	 0	 2	 10	 0.0	 0.0	 0.0
I-genre	 2	 0	 0	 1.0	 1.0	 1.0
B-state	 55	 0	 4	 1.0	 0.9322034	 0.9649123
I-object_name	 383	 29	 16	 0.9296116	 0.9598997	 0.9445129
B-current_location	 13	 0	 1	 1.0	 0.9285714	 0.9629629
B-timeRange	 102	 8	 5	 0.92727274	 0.95327103	 0.9400922
B-sort	 29	 1	 3	 0.96666664	 0.90625	 0.9354838
I-timeRange	 144	 7	 0	 0.95364237	 1.0	 0.97627115
B-rating_unit	 40	 0	 0	 1.0	 1.0	 1.0
I-current_location	 7	 0	 0	 1.0	 1.0	 1.0
I-state	 6	 0	 0	 1.0	 1.0	 1.0
I-album	 4	 1	 17	 0.8	 0.1904762	 0.30769232
B-entity_name	 31	 4	 2	 0.8857143	 0.93939394	 0.9117647
B-object_name	 134	 22	 13	 0.85897434	 0.91156465	 0.88448846
B-playlist_owner	 70	 1	 0	 0.9859155	 1.0	 0.9929078
I-music_item	 5	 0	 0	 1.0	 1.0	 1.0
I-spatial_relation	 41	 2	 1	 0.95348835	 0.97619045	 0.9647058
I-country	 25	 1	 0	 0.96153843	 1.0	 0.98039216
B-rating_value	 80	 0	 0	 1.0	 1.0	 1.0
B-restaurant_type	 64	 0	 1	 1.0	 0.9846154	 0.9922481
I-playlist_owner	 7	 0	 0	 1.0	 1.0	 1.0
I-cuisine	 1	 0	 0	 1.0	 1.0	 1.0
B-track	 7	 10	 2	 0.4117647	 0.7777778	 0.5384615
B-movie_name	 37	 2	 10	 0.94871795	 0.78723407	 0.8604651
B-party_size_number	 50	 0	 0	 1.0	 1.0	 1.0
I-restaurant_type	 7	 0	 0	 1.0	 1.0	 1.0
B-year	 24	 1	 0	 0.96	 1.0	 0.9795918
B-location_name	 23	 0	 1	 1.0	 0.9583333	 0.9787234
B-object_part_of_series_type	 11	 1	 0	 0.9166667	 1.0	 0.95652175
B-country	 43	 4	 1	 0.9148936	 0.97727275	 0.94505495
I-playlist	 218	 4	 13	 0.981982	 0.94372296	 0.96247244
I-served_dish	 2	 1	 2	 0.6666667	 0.5	 0.57142854
I-track	 19	 29	 2	 0.39583334	 0.9047619	 0.5507246
B-artist	 99	 4	 8	 0.9611651	 0.92523366	 0.9428571
B-best_rating	 43	 0	 0	 1.0	 1.0	 1.0
I-restaurant_name	 35	 2	 1	 0.9459459	 0.9722222	 0.9589041
B-object_select	 40	 1	 0	 0.9756098	 1.0	 0.9876543
B-cuisine	 12	 1	 2	 0.9230769	 0.85714287	 0.8888889
B-movie_type	 33	 0	 0	 1.0	 1.0	 1.0
I-geographic_poi	 33	 0	 0	 1.0	 1.0	 1.0
tp: 3121 fp: 177 fn: 155 labels: 69
Macro-average	 prec: 0.91982585, rec: 0.9205297, f1: 0.9201776
Micro-average	 prec: 0.9463311, rec: 0.9526862, f1: 0.949498
```