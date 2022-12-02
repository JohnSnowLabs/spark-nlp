{%- capture title -%}
WordEmbeddings
{%- endcapture -%}

{%- capture model_description -%}
Word Embeddings lookup annotator that maps tokens to vectors

This is the instantiated model of WordEmbeddings.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = WordEmbeddingsModel.pretrained()
    .setInputCols("document", "token")
    .setOutputCol("embeddings")
```
The default model is `"glove_100d"`, if no name is provided.
For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models?task=Embeddings).

There are also two convenient functions to retrieve the embeddings coverage with respect to the transformed dataset:
  - `withCoverageColumn(dataset, embeddingsCol, outputCol)`:
    Adds a custom column with word coverage stats for the embedded field:
    (`coveredWords`, `totalWords`, `coveragePercentage`). This creates a new column with statistics for each row.
    ```
    val wordsCoverage = WordEmbeddingsModel.withCoverageColumn(resultDF, "embeddings", "cov_embeddings")
    wordsCoverage.select("text","cov_embeddings").show(false)
    +-------------------+--------------+
    |text               |cov_embeddings|
    +-------------------+--------------+
    |This is a sentence.|[5, 5, 1.0]   |
    +-------------------+--------------+
    ```
  - `overallCoverage(dataset, embeddingsCol)`:
    Calculates overall word coverage for the whole data in the embedded field.
    This returns a single coverage object considering all rows in the field.
    ```
    val wordsOverallCoverage = WordEmbeddingsModel.overallCoverage(wordsCoverage,"embeddings").percentage
    1.0
    ```

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb)
and the [WordEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddingsTestSpec.scala).
{%- endcapture -%}

{%- capture model_input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture model_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

embeddings = WordEmbeddingsModel.pretrained() \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      embeddings,
      embeddingsFinisher
    ])

data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-0.570580005645752,0.44183000922203064,0.7010200023651123,-0.417129993438720...|
|[-0.542639970779419,0.4147599935531616,1.0321999788284302,-0.4024400115013122...|
|[-0.2708599865436554,0.04400600120425224,-0.020260000601410866,-0.17395000159...|
|[0.6191999912261963,0.14650000631809235,-0.08592499792575836,-0.2629800140857...|
|[-0.3397899866104126,0.20940999686717987,0.46347999572753906,-0.6479200124740...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val embeddings = WordEmbeddingsModel.pretrained()
  .setInputCols("document", "token")
  .setOutputCol("embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    embeddings,
    embeddingsFinisher
  ))

val data = Seq("This is a sentence.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-0.570580005645752,0.44183000922203064,0.7010200023651123,-0.417129993438720...|
|[-0.542639970779419,0.4147599935531616,1.0321999788284302,-0.4024400115013122...|
|[-0.2708599865436554,0.04400600120425224,-0.020260000601410866,-0.17395000159...|
|[0.6191999912261963,0.14650000631809235,-0.08592499792575836,-0.2629800140857...|
|[-0.3397899866104126,0.20940999686717987,0.46347999572753906,-0.6479200124740...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[WordEmbeddingsModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/WordEmbeddingsModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[WordEmbeddingsModel](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/word_embeddings/index.html#sparknlp.annotator.embeddings.word_embeddings.WordEmbeddingsModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[WordEmbeddingsModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddingsModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Word Embeddings lookup annotator that maps tokens to vectors.

For instantiated/pretrained models, see WordEmbeddingsModel.

A custom token lookup dictionary for embeddings can be set with `setStoragePath`.
Each line of the provided file needs to have a token, followed by their vector representation, delimited by a spaces.
```
...
are 0.39658191506190343 0.630968081620067 0.5393722253731201 0.8428180123359783
were 0.7535235923631415 0.9699218875629833 0.10397182122983872 0.11833962569383116
stress 0.0492683418305907 0.9415954572751959 0.47624463167525755 0.16790967216778263
induced 0.1535748762292387 0.33498936903209897 0.9235178224122094 0.1158772920395934
...
```
If a token is not found in the dictionary, then the result will be a zero vector of the same dimension.
Statistics about the rate of converted tokens, can be retrieved with`[WordEmbeddingsModel.withCoverageColumn`
and `WordEmbeddingsModel.overallCoverage`.

For extended examples of usage, see the [Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/3.SparkNLP_Pretrained_Models.ipynb)
and the [WordEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddingsTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
DOCUMENT, TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
# In this example, the file `random_embeddings_dim4.txt` has the form of the content above.

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

embeddings = WordEmbeddings() \
    .setStoragePath("src/test/resources/random_embeddings_dim4.txt", ReadAs.TEXT) \
    .setStorageRef("glove_4d") \
    .setDimension(4) \
    .setInputCols(["document", "token"]) \
    .setOutputCol("embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      embeddings,
      embeddingsFinisher
    ])

data = spark.createDataFrame([["The patient was diagnosed with diabetes."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(truncate=False)
+----------------------------------------------------------------------------------+
|result                                                                            |
+----------------------------------------------------------------------------------+
|[0.9439099431037903,0.4707513153553009,0.806300163269043,0.16176554560661316]     |
|[0.7966810464859009,0.5551124811172485,0.8861005902290344,0.28284206986427307]    |
|[0.025029370561242104,0.35177749395370483,0.052506182342767715,0.1887107789516449]|
|[0.08617766946554184,0.8399239182472229,0.5395117998123169,0.7864698767662048]    |
|[0.6599600911140442,0.16109347343444824,0.6041093468666077,0.8913561105728149]    |
|[0.5955275893211365,0.01899011991918087,0.4397728443145752,0.8911281824111938]    |
|[0.9840458631515503,0.7599489092826843,0.9417727589607239,0.8624503016471863]     |
+----------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
// In this example, the file `random_embeddings_dim4.txt` has the form of the content above.
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.WordEmbeddings
import com.johnsnowlabs.nlp.util.io.ReadAs
import com.johnsnowlabs.nlp.EmbeddingsFinisher
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val embeddings = new WordEmbeddings()
  .setStoragePath("src/test/resources/random_embeddings_dim4.txt", ReadAs.TEXT)
  .setStorageRef("glove_4d")
  .setDimension(4)
  .setInputCols("document", "token")
  .setOutputCol("embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    embeddings,
    embeddingsFinisher
  ))

val data = Seq("The patient was diagnosed with diabetes.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(false)
+----------------------------------------------------------------------------------+
|result                                                                            |
+----------------------------------------------------------------------------------+
|[0.9439099431037903,0.4707513153553009,0.806300163269043,0.16176554560661316]     |
|[0.7966810464859009,0.5551124811172485,0.8861005902290344,0.28284206986427307]    |
|[0.025029370561242104,0.35177749395370483,0.052506182342767715,0.1887107789516449]|
|[0.08617766946554184,0.8399239182472229,0.5395117998123169,0.7864698767662048]    |
|[0.6599600911140442,0.16109347343444824,0.6041093468666077,0.8913561105728149]    |
|[0.5955275893211365,0.01899011991918087,0.4397728443145752,0.8911281824111938]    |
|[0.9840458631515503,0.7599489092826843,0.9417727589607239,0.8624503016471863]     |
+----------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[WordEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/WordEmbeddings)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[WordEmbeddings](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/word_embeddings/index.html#sparknlp.annotator.embeddings.word_embeddings.WordEmbeddings)
{%- endcapture -%}

{%- capture approach_source_link -%}
[WordEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddings.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_api_link=model_python_api_link
model_api_link=model_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_python_api_link=approach_python_api_link
approach_api_link=approach_api_link
approach_source_link=approach_source_link
%}
