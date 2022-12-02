{%- capture title -%}
ChunkTokenizer
{%- endcapture -%}

{%- capture model_description -%}
Instantiated model of the ChunkTokenizer.
For usage and examples see the documentation of the main class.
{%- endcapture -%}

{%- capture model_input_anno -%}
CHUNK
{%- endcapture -%}

{%- capture model_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_api_link -%}
[ChunkTokenizerModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ChunkTokenizerModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[ChunkTokenizerModel](/api/python/reference/autosummary/python/sparknlp/annotator/token/chunk_tokenizer/index.html#sparknlp.annotator.token.chunk_tokenizer.ChunkTokenizerModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[ChunkTokenizerModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ChunkTokenizerModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Tokenizes and flattens extracted NER chunks.

The ChunkTokenizer will split the extracted NER `CHUNK` type Annotations and will create `TOKEN` type Annotations.
The result is then flattened, resulting in a single array.

For extended examples of usage, see the [ChunkTokenizerTestSpec](https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/ChunkTokenizerTestSpec.scala).
{%- endcapture -%}

{%- capture approach_input_anno -%}
CHUNK
{%- endcapture -%}

{%- capture approach_output_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_python_example -%}
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline

documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")

tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")

entityExtractor = TextMatcher() \
    .setInputCols(["sentence", "token"]) \
    .setEntities("src/test/resources/entity-extractor/test-chunks.txt", ReadAs.TEXT) \
    .setOutputCol("entity")

chunkTokenizer = ChunkTokenizer() \
    .setInputCols(["entity"]) \
    .setOutputCol("chunk_token")

pipeline = Pipeline().setStages([
      documentAssembler,
      sentenceDetector,
      tokenizer,
      entityExtractor,
      chunkTokenizer
    ])

data = spark.createDataFrame([[
    "Hello world, my name is Michael, I am an artist and I work at Benezar",
    "Robert, an engineer from Farendell, graduated last year. The other one, Lucas, graduated last week."
]]).toDF("text")
result = pipeline.fit(data).transform(data)

result.selectExpr("entity.result as entity" , "chunk_token.result as chunk_token").show(truncate=False)
+-----------------------------------------------+---------------------------------------------------+
|entity                                         |chunk_token                                        |
+-----------------------------------------------+---------------------------------------------------+
|[world, Michael, work at Benezar]              |[world, Michael, work, at, Benezar]                |
|[engineer from Farendell, last year, last week]|[engineer, from, Farendell, last, year, last, week]|
+-----------------------------------------------+---------------------------------------------------+

{%- endcapture -%}

{%- capture approach_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.{ChunkTokenizer, TextMatcher, Tokenizer}
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.util.io.ReadAs
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val sentenceDetector = new SentenceDetector()
  .setInputCols(Array("document"))
  .setOutputCol("sentence")

val tokenizer = new Tokenizer()
  .setInputCols(Array("sentence"))
  .setOutputCol("token")

val entityExtractor = new TextMatcher()
  .setInputCols("sentence", "token")
  .setEntities("src/test/resources/entity-extractor/test-chunks.txt", ReadAs.TEXT)
  .setOutputCol("entity")

val chunkTokenizer = new ChunkTokenizer()
  .setInputCols("entity")
  .setOutputCol("chunk_token")

val pipeline = new Pipeline().setStages(Array(
    documentAssembler,
    sentenceDetector,
    tokenizer,
    entityExtractor,
    chunkTokenizer
  ))

val data = Seq(
  "Hello world, my name is Michael, I am an artist and I work at Benezar",
  "Robert, an engineer from Farendell, graduated last year. The other one, Lucas, graduated last week."
).toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("entity.result as entity" , "chunk_token.result as chunk_token").show(false)
+-----------------------------------------------+---------------------------------------------------+
|entity                                         |chunk_token                                        |
+-----------------------------------------------+---------------------------------------------------+
|[world, Michael, work at Benezar]              |[world, Michael, work, at, Benezar]                |
|[engineer from Farendell, last year, last week]|[engineer, from, Farendell, last, year, last, week]|
+-----------------------------------------------+---------------------------------------------------+

{%- endcapture -%}

{%- capture approach_api_link -%}
[ChunkTokenizer](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/annotators/ChunkTokenizer)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[ChunkTokenizer](/api/python/reference/autosummary/python/sparknlp/annotator/token/chunk_tokenizer/index.html#sparknlp.annotator.token.chunk_tokenizer.ChunkTokenizer)
{%- endcapture -%}

{%- capture approach_source_link -%}
[ChunkTokenizer](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/annotators/ChunkTokenizer.scala)
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
