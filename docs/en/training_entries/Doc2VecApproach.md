{%- capture title -%}
Doc2VecApproach
{%- endcapture -%}

{%- capture description -%}
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

{%- capture input_anno -%}
TOKEN
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

{%- capture scala_example -%}
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

{%- capture api_link -%}
[Doc2VecApproach](https://nlp.johnsnowlabs.com/api/com/johnsnowlabs/nlp/embeddings/Doc2VecApproach)
{%- endcapture -%}

{%- capture python_api_link -%}
[Doc2VecApproach](/api/python/reference/autosummary/python/sparknlp/annotator/embeddings/doc2vec/index.html#sparknlp.annotator.embeddings.doc2vec.Doc2VecApproach)
{%- endcapture -%}

{%- capture source_link -%}
[Doc2VecApproach](https://github.com/JohnSnowLabs/spark-nlp/tree/master/src/main/scala/com/johnsnowlabs/nlp/embeddings/Doc2VecApproach.scala)
{%- endcapture -%}

{% include templates/training_anno_template.md
title=title
description=description
input_anno=input_anno
output_anno=output_anno
python_example=python_example
scala_example=scala_example
python_api_link=python_api_link
api_link=api_link
source_link=source_link
%}