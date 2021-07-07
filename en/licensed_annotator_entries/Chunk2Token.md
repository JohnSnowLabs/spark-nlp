{%- capture title -%}
Chunk2Token
{%- endcapture -%}

{%- capture description -%}
A feature transformer that converts the input array of strings (annotatorType CHUNK) into an
array of chunk-based tokens (annotatorType TOKEN).

When the input is empty, an empty array is returned.

This Annotator is specially convenient when using NGramGenerator annotations as inputs to WordEmbeddingsModels
{%- endcapture -%}

{%- capture input_anno -%}
CHUNK
{%- endcapture -%}

{%- capture output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import *
import sparknlp_jsl
from sparknlp_jsl.base import *
from sparknlp_jsl.annotator import *
from pyspark.ml import Pipeline
# Define a pipeline for generating n-grams
data = spark.createDataFrame([["A 63-year-old man presents to the hospital ..."]]).toDF("text")
document = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentenceDetector = SentenceDetector().setInputCols(["document"]).setOutputCol("sentence")
token = Tokenizer().setInputCols(["sentence"]).setOutputCol("token")
ngrammer = NGramGenerator() \
 .setN(2) \
 .setEnableCumulative(False) \
 .setInputCols(["token"]) \
 .setOutputCol("ngrams") \
 .setDelimiter("_")

# Stage to convert n-gram CHUNKS to TOKEN type
chunk2Token = Chunk2Token().setInputCols(["ngrams"]).setOutputCol("ngram_tokens")
trainingPipeline = Pipeline(stages=[document, sentenceDetector, token, ngrammer, chunk2Token]).fit(data)

result = trainingPipeline.transform(data).cache()
result.selectExpr("explode(ngram_tokens)").show(5, False)
    +----------------------------------------------------------------+
    |col                                                             |
    +----------------------------------------------------------------+
    |{token, 3, 15, A_63-year-old, {sentence -> 0, chunk -> 0}, []}  |
    |{token, 5, 19, 63-year-old_man, {sentence -> 0, chunk -> 1}, []}|
    |{token, 17, 28, man_presents, {sentence -> 0, chunk -> 2}, []}  |
    |{token, 21, 31, presents_to, {sentence -> 0, chunk -> 3}, []}   |
    |{token, 30, 35, to_the, {sentence -> 0, chunk -> 4}, []}        |
    +----------------------------------------------------------------+

{%- endcapture -%}

{%- capture scala_example -%}
// Define a pipeline for generating n-grams
val data = Seq(("A 63-year-old man presents to the hospital ...")).toDF("text")
val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
val sentenceDetector = new SentenceDetector().setInputCols("document").setOutputCol("sentence")
val token = new Tokenizer().setInputCols("sentence").setOutputCol("token")
val ngrammer = new NGramGenerator()
 .setN(2)
 .setEnableCumulative(false)
 .setInputCols("token")
 .setOutputCol("ngrams")
 .setDelimiter("_")

// Stage to convert n-gram CHUNKS to TOKEN type
val chunk2Token = new Chunk2Token().setInputCols("ngrams").setOutputCol("ngram_tokens")
val trainingPipeline = new Pipeline().setStages(Array(document, sentenceDetector, token, ngrammer, chunk2Token)).fit(data)

val result = trainingPipeline.transform(data).cache()
result.selectExpr("explode(ngram_tokens)").show(5, false)
+----------------------------------------------------------------+
|col                                                             |
+----------------------------------------------------------------+
|{token, 3, 15, A_63-year-old, {sentence -> 0, chunk -> 0}, []}  |
|{token, 5, 19, 63-year-old_man, {sentence -> 0, chunk -> 1}, []}|
|{token, 17, 28, man_presents, {sentence -> 0, chunk -> 2}, []}  |
|{token, 21, 31, presents_to, {sentence -> 0, chunk -> 3}, []}   |
|{token, 30, 35, to_the, {sentence -> 0, chunk -> 4}, []}        |
+----------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[Chunk2Token](https://nlp.johnsnowlabs.com/licensed/api/com/johnsnowlabs/nlp/annotators/Chunk2Token)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
api_link=api_link%}