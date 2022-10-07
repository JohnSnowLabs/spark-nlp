{%- capture title -%}
SentenceEmbeddings
{%- endcapture -%}

{%- capture description -%}
Converts the results from WordEmbeddings, BertEmbeddings, or ElmoEmbeddings into sentence
or document embeddings by either summing up or averaging all the word embeddings in a sentence or a document
(depending on the inputCols).

This can be configured with `setPoolingStrategy`, which either be `"AVERAGE"` or `"SUM"`.

For more extended examples see the
[Spark NLP Workshop](https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Public/5.1_Text_classification_examples_in_SparkML_SparkNLP.ipynb).
and the [SentenceEmbeddingsTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/SentenceEmbeddingsTestSpec.scala).

**TIP:** Here is how you can explode and convert these embeddings into `Vectors` or what's known as `Feature` column so it can be used in Spark ML regression or clustering functions:

<div class="tabs-box tabs-new" markdown="1">

{% include programmingLanguageSelectScalaPython.html %}

```python
from org.apache.spark.ml.linal import Vector, Vectors
from pyspark.sql.functions import udf
# Let's create a UDF to take array of embeddings and output Vectors
@udf(Vector)
def convertToVectorUDF(matrix):
    return Vectors.dense(matrix.toArray.map(_.toDouble))


# Now let's explode the sentence_embeddings column and have a new feature column for Spark ML
pipelineDF.select(explode("sentence_embeddings.embeddings").as("sentence_embedding"))
.withColumn("features", convertToVectorUDF("sentence_embedding"))
```

```scala
import org.apache.spark.ml.linalg.{Vector, Vectors}

// Let's create a UDF to take array of embeddings and output Vectors
val convertToVectorUDF = udf((matrix : Seq[Float]) => {
    Vectors.dense(matrix.toArray.map(_.toDouble))
})

// Now let's explode the sentence_embeddings column and have a new feature column for Spark ML
pipelineDF.select(explode($"sentence_embeddings.embeddings").as("sentence_embedding"))
.withColumn("features", convertToVectorUDF($"sentence_embedding"))
```
</div>
{%- endcapture -%}

{%- capture input_anno -%}
DOCUMENT, WORD_EMBEDDINGS
{%- endcapture -%}

{%- capture output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture python_example -%}
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

embeddingsSentence = SentenceEmbeddings() \
    .setInputCols(["document", "embeddings"]) \
    .setOutputCol("sentence_embeddings") \
    .setPoolingStrategy("AVERAGE")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["sentence_embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True) \
    .setCleanAnnotations(False)

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      embeddings,
      embeddingsSentence,
      embeddingsFinisher
    ])

data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-0.22093398869037628,0.25130119919776917,0.41810303926467896,-0.380883991718...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture note -%}
If you choose `document` as your input for `Tokenizer`, `WordEmbeddings`/`BertEmbeddings`, and `SentenceEmbeddings` then it averages/sums all the embeddings into one array of embeddings. However, if you choose `sentence` as `inputCols` then for each sentence `SentenceEmbeddings` generates one array of embeddings.
{%- endcapture -%}

{%- capture scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings
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

val embeddingsSentence = new SentenceEmbeddings()
  .setInputCols(Array("document", "embeddings"))
  .setOutputCol("sentence_embeddings")
  .setPoolingStrategy("AVERAGE")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("sentence_embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)
  .setCleanAnnotations(false)

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    embeddings,
    embeddingsSentence,
    embeddingsFinisher
  ))

val data = Seq("This is a sentence.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[-0.22093398869037628,0.25130119919776917,0.41810303926467896,-0.380883991718...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture api_link -%}
[SentenceEmbeddings](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/SentenceEmbeddings)
{%- endcapture -%}

{%- capture python_api_link -%}
[SentenceEmbeddings](/api/python/reference/autosummary/sparknlp/annotator/embeddings/sentence_embeddings/index.html#sparknlp.annotator.embeddings.sentence_embeddings.SentenceEmbeddings)
{%- endcapture -%}

{%- capture source_link -%}
[SentenceEmbeddings](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/SentenceEmbeddings.scala)
{%- endcapture -%}

{% include templates/anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
note=note
%}