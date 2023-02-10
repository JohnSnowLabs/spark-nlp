/*
 * Copyright 2017-2022 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp

import com.johnsnowlabs.nlp.util.FinisherUtil
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.param.{BooleanParam, ParamMap, StringArrayParam}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Dataset, Row}

/** Extracts embeddings from Annotations into a more easily usable form.
  *
  * This is useful for example: [[com.johnsnowlabs.nlp.embeddings.WordEmbeddings WordEmbeddings]],
  * [[com.johnsnowlabs.nlp.embeddings.BertEmbeddings BertEmbeddings]],
  * [[com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings SentenceEmbeddings]] and
  * [[com.johnsnowlabs.nlp.embeddings.ChunkEmbeddings ChunkEmbeddings]].
  *
  * By using `EmbeddingsFinisher` you can easily transform your embeddings into array of floats or
  * vectors which are compatible with Spark ML functions such as LDA, K-mean, Random Forest
  * classifier or any other functions that require `featureCol`.
  *
  * For more extended examples see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/annotation/text/english/text-similarity/Spark_NLP_Spark_ML_Text_Similarity.ipynb Examples]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import org.apache.spark.ml.Pipeline
  * import com.johnsnowlabs.nlp.{DocumentAssembler, EmbeddingsFinisher}
  * import com.johnsnowlabs.nlp.annotator.{Normalizer, StopWordsCleaner, Tokenizer, WordEmbeddingsModel}
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols("document")
  *   .setOutputCol("token")
  *
  * val normalizer = new Normalizer()
  *   .setInputCols("token")
  *   .setOutputCol("normalized")
  *
  * val stopwordsCleaner = new StopWordsCleaner()
  *   .setInputCols("normalized")
  *   .setOutputCol("cleanTokens")
  *   .setCaseSensitive(false)
  *
  * val gloveEmbeddings = WordEmbeddingsModel.pretrained()
  *   .setInputCols("document", "cleanTokens")
  *   .setOutputCol("embeddings")
  *   .setCaseSensitive(false)
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("embeddings")
  *   .setOutputCols("finished_sentence_embeddings")
  *   .setOutputAsVector(true)
  *   .setCleanAnnotations(false)
  *
  * val data = Seq("Spark NLP is an open-source text processing library.")
  *   .toDF("text")
  * val pipeline = new Pipeline().setStages(Array(
  *   documentAssembler,
  *   tokenizer,
  *   normalizer,
  *   stopwordsCleaner,
  *   gloveEmbeddings,
  *   embeddingsFinisher
  * )).fit(data)
  *
  * val result = pipeline.transform(data)
  * val resultWithSize = result.selectExpr("explode(finished_sentence_embeddings)")
  *   .map { row =>
  *     val vector = row.getAs[org.apache.spark.ml.linalg.DenseVector](0)
  *     (vector.size, vector)
  *   }.toDF("size", "vector")
  *
  * resultWithSize.show(5, 80)
  * +----+--------------------------------------------------------------------------------+
  * |size|                                                                          vector|
  * +----+--------------------------------------------------------------------------------+
  * | 100|[0.1619900017976761,0.045552998781204224,-0.03229299932718277,-0.685609996318...|
  * | 100|[-0.42416998744010925,1.1378999948501587,-0.5717899799346924,-0.5078899860382...|
  * | 100|[0.08621499687433243,-0.15772999823093414,-0.06067200005054474,0.395359992980...|
  * | 100|[-0.4970499873161316,0.7164199948310852,0.40119001269340515,-0.05761000141501...|
  * | 100|[-0.08170200139284134,0.7159299850463867,-0.20677000284194946,0.0295659992843...|
  * +----+--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[com.johnsnowlabs.nlp.Finisher Finisher]] for finishing Strings
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class EmbeddingsFinisher(override val uid: String)
    extends Transformer
    with DefaultParamsWritable {

  /** Name of input annotation cols containing embeddings
    *
    * @group param
    */
  val inputCols: StringArrayParam =
    new StringArrayParam(this, "inputCols", "Name of input annotation cols containing embeddings")

  /** Name of EmbeddingsFinisher output cols
    *
    * @group param
    */
  val outputCols: StringArrayParam =
    new StringArrayParam(this, "outputCols", "Name of EmbeddingsFinisher output cols")

  /** Whether to remove all the existing annotation columns (Default: `true`)
    *
    * @group param
    */
  val cleanAnnotations: BooleanParam =
    new BooleanParam(
      this,
      "cleanAnnotations",
      "Whether to remove all the existing annotation columns (Default: `true`)")

  /** If enabled it will output the embeddings as Vectors instead of arrays (Default: `false`)
    *
    * @group param
    */
  val outputAsVector: BooleanParam =
    new BooleanParam(
      this,
      "outputAsVector",
      "If enabled it will output the embeddings as Vectors instead of arrays (Default: `false`)")

  /** Name of input annotation cols containing embeddings
    *
    * @group setParam
    */
  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  /** Name of input annotation cols containing embeddings
    *
    * @group setParam
    */
  def setInputCols(value: String*): this.type = setInputCols(value.toArray)

  /** Name of EmbeddingsFinisher output cols
    *
    * @group setParam
    */
  def setOutputCols(value: Array[String]): this.type = set(outputCols, value)

  /** Name of EmbeddingsFinisher output cols
    *
    * @group setParam
    */
  def setOutputCols(value: String*): this.type = setOutputCols(value.toArray)

  /** Whether to remove all the existing annotation columns (Default: `true`)
    *
    * @group setParam
    */
  def setCleanAnnotations(value: Boolean): this.type = set(cleanAnnotations, value)

  /** If enabled it will output the embeddings as Vectors instead of arrays (Default: `false`)
    *
    * @group setParam
    */
  def setOutputAsVector(value: Boolean): this.type = set(outputAsVector, value)

  /** Name of input annotation cols containing embeddings
    *
    * @group getParam
    */
  def getOutputCols: Array[String] = get(outputCols).getOrElse(getInputCols.map("finished_" + _))

  /** Name of EmbeddingsFinisher output cols
    *
    * @group getParam
    */
  def getInputCols: Array[String] = $(inputCols)

  /** Whether to remove all the existing annotation columns (Default: `true`)
    *
    * @group getParam
    */
  def getCleanAnnotations: Boolean = $(cleanAnnotations)

  /** If enabled it will output the embeddings as Vectors instead of arrays (Default: `false`)
    *
    * @group getParam
    */
  def getOutputAsVector: Boolean = $(outputAsVector)

  setDefault(cleanAnnotations -> true, outputAsVector -> false)

  def this() = this(Identifiable.randomUID("embeddings_finisher"))

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = {

    require(
      getInputCols.length == getOutputCols.length,
      "inputCols and outputCols length must match")

    val embeddingsAnnotators =
      Seq(AnnotatorType.WORD_EMBEDDINGS, AnnotatorType.SENTENCE_EMBEDDINGS)

    getInputCols.foreach { annotationColumn =>
      FinisherUtil.checkIfInputColsExist(getInputCols, schema)
      FinisherUtil.checkIfAnnotationColumnIsSparkNLPAnnotation(schema, annotationColumn)

      /** Check if the annotationColumn has embeddings It must be at least of one these
        * annotators: WordEmbeddings, BertEmbeddings, ChunkEmbeddings, or SentenceEmbeddings
        */
      require(
        embeddingsAnnotators.contains(
          schema(annotationColumn).metadata.getString("annotatorType")),
        s"column [$annotationColumn] must be a type of either WordEmbeddings, BertEmbeddings, ChunkEmbeddings, or SentenceEmbeddings")

    }
    val metadataFields = FinisherUtil.getMetadataFields(getOutputCols, $(outputAsVector))

    val outputFields = schema.fields ++
      getOutputCols.map(outputCol => {
        if ($(outputAsVector))
          StructField(outputCol, ArrayType(VectorType), nullable = false)
        else
          StructField(outputCol, ArrayType(FloatType), nullable = false)
      }) ++ metadataFields

    val cleanFields = FinisherUtil.getCleanFields($(cleanAnnotations), outputFields)

    StructType(cleanFields)
  }

  private def vectorsAsArray: UserDefinedFunction = udf { embeddings: Seq[Seq[Float]] =>
    embeddings
  }

  private def vectorsAsVectorType: UserDefinedFunction = udf { embeddings: Seq[Seq[Float]] =>
    embeddings.map(embedding => Vectors.dense(embedding.toArray.map(_.toDouble)))
  }

  override def transform(dataset: Dataset[_]): Dataset[Row] = {
    require(
      getInputCols.length == getOutputCols.length,
      "inputCols and outputCols length must match")

    val cols = getInputCols.zip(getOutputCols)
    var flattened = dataset
    cols.foreach { case (inputCol, outputCol) =>
      flattened = {
        flattened.withColumn(
          outputCol, {
            if ($(outputAsVector))
              vectorsAsVectorType(flattened.col(inputCol + ".embeddings"))
            else
              vectorsAsArray(flattened.col(inputCol + ".embeddings"))
          })
      }
    }

    FinisherUtil.cleaningAnnotations($(cleanAnnotations), flattened.toDF())
  }

}

/** This is the companion object of [[EmbeddingsFinisher]]. Please refer to that class for the
  * documentation.
  */
object EmbeddingsFinisher extends DefaultParamsReadable[EmbeddingsFinisher]
