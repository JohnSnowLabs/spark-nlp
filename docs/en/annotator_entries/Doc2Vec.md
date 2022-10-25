{%- capture title -%}
Doc2Vec 1
{%- endcapture -%}

{%- capture model_description -%}
Word2Vec model that creates vector representations of words in a text corpus.

The algorithm first constructs a vocabulary from the corpus
and then learns vector representation of words in the vocabulary.
The vector representation can be used as features in
natural language processing and machine learning algorithms.

We use Word2Vec implemented in Spark ML. It uses skip-gram model in our implementation and a hierarchical softmax
method to train the model. The variable names in the implementation match the original C implementation.

This is the instantiated model of the Doc2VecApproach.
For training your own model, please see the documentation of that class.

Pretrained models can be loaded with `pretrained` of the companion object:
```
val embeddings = Doc2VecModel.pretrained()
  .setInputCols("token")
  .setOutputCol("embeddings")
```
The default model is `"doc2vec_gigaword_300"`, if no name is provided.

For available pretrained models please see the [Models Hub](https://nlp.johnsnowlabs.com/models).

**Sources** :

For the original C implementation, see [https://code.google.com/p/word2vec/](https://code.google.com/p/word2vec/)

For the research paper, see
[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
and [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546v1.pdf).
{%- endcapture -%}

{%- capture model_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture model_output_anno -%}
SENTENCE_EMBEDDINGS
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

embeddings = Doc2VecModel.pretrained() \
    .setInputCols(["token"]) \
    .setOutputCol("embeddings")

embeddingsFinisher = EmbeddingsFinisher() \
    .setInputCols(["embeddings"]) \
    .setOutputCols("finished_embeddings") \
    .setOutputAsVector(True)

pipeline = Pipeline().setStages([
    documentAssembler,
    tokenizer,
    embeddings,
    embeddingsFinisher
])

data = spark.createDataFrame([["This is a sentence."]]).toDF("text")
result = pipeline.fit(data).transform(data)
result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.06222493574023247,0.011579325422644615,0.009919632226228714,0.109361454844...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture model_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.{Tokenizer, Doc2VecModel}
import com.johnsnowlabs.nlp.EmbeddingsFinisher

import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val embeddings = Doc2VecModel.pretrained()
  .setInputCols("token")
  .setOutputCol("embeddings")

val embeddingsFinisher = new EmbeddingsFinisher()
  .setInputCols("embeddings")
  .setOutputCols("finished_embeddings")
  .setOutputAsVector(true)

val pipeline = new Pipeline().setStages(Array(
  documentAssembler,
  tokenizer,
  embeddings,
  embeddingsFinisher
))

val data = Seq("This is a sentence.").toDF("text")
val result = pipeline.fit(data).transform(data)

result.selectExpr("explode(finished_embeddings) as result").show(1, 80)
+--------------------------------------------------------------------------------+
|                                                                          result|
+--------------------------------------------------------------------------------+
|[0.06222493574023247,0.011579325422644615,0.009919632226228714,0.109361454844...|
+--------------------------------------------------------------------------------+

{%- endcapture -%}

{%- capture model_api_link -%}
[Doc2VecModel](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/Doc2VecModel)
{%- endcapture -%}

{%- capture model_python_api_link -%}
[Doc2VecModel](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/doc2vec/index.html#sparknlp.annotator.embeddings.doc2vec.Doc2VecModel)
{%- endcapture -%}

{%- capture model_source_link -%}
[Doc2VecModel](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/Doc2VecModel.scala)
{%- endcapture -%}

{%- capture approach_description -%}
Trains a Word2Vec model that creates vector representations of words in a text corpus.

The algorithm first constructs a vocabulary from the corpus
and then learns vector representation of words in the vocabulary.
The vector representation can be used as features in
natural language processing and machine learning algorithms.

We use Word2Vec implemented in Spark ML. It uses skip-gram model in our implementation and a hierarchical softmax
method to train the model. The variable names in the implementation match the original C implementation.

For instantiated/pretrained models, see Doc2VecModel.

**Sources** :

For the original C implementation, see [https://code.google.com/p/word2vec/](https://code.google.com/p/word2vec/)

For the research paper, see
[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
and [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/pdf/1310.4546v1.pdf).
{%- endcapture -%}

{%- capture approach_input_anno -%}
TOKEN
{%- endcapture -%}

{%- capture approach_output_anno -%}
SENTENCE_EMBEDDINGS
{%- endcapture -%}

{%- capture approach_python_example -%}
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

embeddings = Doc2VecApproach() \
    .setInputCols(["token"]) \
    .setOutputCol("embeddings")

pipeline = Pipeline() \
    .setStages([
      documentAssembler,
      tokenizer,
      embeddings
    ])

path = "sherlockholmes.txt"
dataset = spark.read.text(path).toDF("text")
pipelineModel = pipeline.fit(dataset)

{%- endcapture -%}

{%- capture approach_scala_example -%}
import spark.implicits._
import com.johnsnowlabs.nlp.annotator.{Tokenizer, Doc2VecApproach}
import com.johnsnowlabs.nlp.base.DocumentAssembler
import org.apache.spark.ml.Pipeline

val documentAssembler = new DocumentAssembler()
  .setInputCol("text")
  .setOutputCol("document")

val tokenizer = new Tokenizer()
  .setInputCols(Array("document"))
  .setOutputCol("token")

val embeddings = new Doc2VecApproach()
  .setInputCols("token")
  .setOutputCol("embeddings")

val pipeline = new Pipeline()
  .setStages(Array(
    documentAssembler,
    tokenizer,
    embeddings
  ))

val path = "src/test/resources/spell/sherlockholmes.txt"
val dataset = spark.sparkContext.textFile(path)
  .toDF("text")
val pipelineModel = pipeline.fit(dataset)

{%- endcapture -%}

{%- capture approach_api_link -%}
[Doc2VecApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/Doc2VecApproach)
{%- endcapture -%}

{%- capture approach_python_api_link -%}
[Doc2VecApproach](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/doc2vec/index.html#sparknlp.annotator.embeddings.doc2vec.Doc2VecApproach)
{%- endcapture -%}

{%- capture approach_source_link -%}
[Doc2VecApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/Doc2VecApproach.scala)
{%- endcapture -%}


{% include templates/approach_model_template.md
title=title
model_description=model_description
model_input_anno=model_input_anno
model_output_anno=model_output_anno
model_python_example=model_python_example
model_scala_example=model_scala_example
model_api_link=model_api_link
model_python_api_link=model_python_api_link
model_source_link=model_source_link
approach_description=approach_description
approach_input_anno=approach_input_anno
approach_output_anno=approach_output_anno
approach_python_example=approach_python_example
approach_scala_example=approach_scala_example
approach_api_link=approach_api_link
approach_python_api_link=approach_python_api_link
approach_source_link=approach_source_link
%}
