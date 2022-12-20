{%- capture title -%}
DateNormalizer
{%- endcapture -%}

{%- capture model -%}
model
{%- endcapture -%}

{%- capture model_description -%}

This annotator transforms date mentions to a common standard format: YYYY/MM/DD. It is useful when using data from different sources, some times from different countries that has different formats to represent dates.

For the relative dates (next year, past month, etc.), you can define an achor date to create the normalized date by setting the parameters `anchorDateYear`, `anchorDateMonth`, and `anchorDateDay`.

The resultant chunk date will contain a metada indicating whether the normalization was successful or not (True / False). 

{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_python_medical -%}

from pyspark.sql.types import StringType

dates = [
    "08/02/2018",
    "11/2018",
    "11/01/2018",
    "12Mar2021",
    "Jan 30, 2018",
    "13.04.1999",
    "3April 2020",
    "next monday",
    "today",
    "next week",
]
df = spark.createDataFrame(dates, StringType()).toDF("original_date")

document_assembler = (
    DocumentAssembler().setInputCol("original_date").setOutputCol("document")
)

doc2chunk = Doc2Chunk().setInputCols("document").setOutputCol("date_chunk")

date_normalizer = (
    DateNormalizer()
    .setInputCols("date_chunk")
    .setOutputCol("date")
    .setAnchorDateYear(2000)
    .setAnchorDateMonth(3)
    .setAnchorDateDay(15)
)

pipeline = Pipeline(stages=[document_assembler, doc2chunk, date_normalizer])

result = pipeline.fit(df).transform(df)
result.selectExpr(
    "date.result as normalized_date",
    "original_date",
    "date.metadata[0].normalized as metadata",
).show()

+---------------+-------------+--------+
|normalized_date|original_date|metadata|
+---------------+-------------+--------+
|   [2018/08/02]|   08/02/2018|    true|
|   [2018/11/DD]|      11/2018|    true|
|   [2018/11/01]|   11/01/2018|    true|
|   [2021/03/12]|    12Mar2021|    true|
|   [2018/01/30]| Jan 30, 2018|    true|
|   [1999/04/13]|   13.04.1999|    true|
|   [2020/04/03]|  3April 2020|    true|
|   [2000/03/20]|  next monday|    true|
|   [2000/03/15]|        today|    true|
|   [2000/03/22]|    next week|    true|
+---------------+-------------+--------+

{%- endcapture -%}


{%- capture model_scala_medical -%}

val df = Seq(("08/02/2018"),("11/2018"),("11/01/2018"),("next monday"),("today"),("next week")).toDF("original_date")

val documentAssembler = new DocumentAssembler().setInputCol("original_date").setOutputCol("document")

val chunksDF = documentAssembler
				  .transform(df)
				  .mapAnnotationsCol[Seq[Annotation]]("document",
													  "chunk_date",
													   CHUNK,
												  (aa:Seq[Annotation]) =>
													aa.map( ann => ann.copy(annotatorType = CHUNK)))
val dateNormalizerModel = new DateNormalizer()
        .setInputCols("chunk_date")
        .setOutputCol("date")
        .setAnchorDateDay(15)
        .setAnchorDateMonth(3)
        .setAnchorDateYear(2000)
val dateDf = dateNormalizerModel.transform(chunksDF)

dateDf.select("chunk_date.result","text").show()
+-------------+-------------+
|       result|original_date|
+-------------+-------------+
| [08/02/2018]|   08/02/2018|
|    [11/2018]|      11/2018|
| [11/01/2018]|   11/01/2018|
|[next monday]|  next monday|
|      [today]|        today|
|  [next week]|    next week|
+-------------+-------------+
{%- endcapture -%}

{%- capture model_api_link -%}
[DateNormalizer](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/normalizer/DateNormalizer.html)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[DateNormalizer](https://nlp.johnsnowlabs.com/licensed/api/python/reference/autosummary/sparknlp_jsl/annotator/normalizer/date_normalizer/index.html#sparknlp_jsl.annotator.normalizer.date_normalizer.DateNormalizer)
{%- endcapture -%}


{% include templates/licensed_approach_model_medical_fin_leg_template.md
title=title
model=model
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_medical=model_python_medical
model_scala_medical=model_scala_medical
model_api_link=model_api_link
model_python_api_link=model_python_api_link
%}
