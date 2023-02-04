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

package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotators.common.WordpieceEmbeddingsSentence
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, HasSimpleAnnotate}
import org.apache.spark.ml.param.{BooleanParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.DataFrame

import scala.collection.Map

object PoolingStrategy {

  object AnnotatorType {
    val AVERAGE = "AVERAGE"
    val SUM = "SUM"
  }

}

/** This annotator utilizes [[WordEmbeddings]], [[BertEmbeddings]] etc. to generate chunk
  * embeddings from either [[com.johnsnowlabs.nlp.annotators.Chunker Chunker]],
  * [[com.johnsnowlabs.nlp.annotators.NGramGenerator NGramGenerator]], or
  * [[com.johnsnowlabs.nlp.annotators.ner.NerConverter NerConverter]] outputs.
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/annotation/text/english/embeddings/ChunkEmbeddings.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/ChunkEmbeddingsTestSpec.scala ChunkEmbeddingsTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
  * import com.johnsnowlabs.nlp.annotators.{NGramGenerator, Tokenizer}
  * import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
  * import com.johnsnowlabs.nlp.embeddings.ChunkEmbeddings
  * import org.apache.spark.ml.Pipeline
  *
  * // Extract the Embeddings from the NGrams
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val sentence = new SentenceDetector()
  *   .setInputCols("document")
  *   .setOutputCol("sentence")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("sentence"))
  *   .setOutputCol("token")
  *
  * val nGrams = new NGramGenerator()
  *   .setInputCols("token")
  *   .setOutputCol("chunk")
  *   .setN(2)
  *
  * val embeddings = WordEmbeddingsModel.pretrained()
  *   .setInputCols("sentence", "token")
  *   .setOutputCol("embeddings")
  *   .setCaseSensitive(false)
  *
  * // Convert the NGram chunks into Word Embeddings
  * val chunkEmbeddings = new ChunkEmbeddings()
  *   .setInputCols("chunk", "embeddings")
  *   .setOutputCol("chunk_embeddings")
  *   .setPoolingStrategy("AVERAGE")
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     sentence,
  *     tokenizer,
  *     nGrams,
  *     embeddings,
  *     chunkEmbeddings
  *   ))
  *
  * val data = Seq("This is a sentence.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(chunk_embeddings) as result")
  *   .select("result.annotatorType", "result.result", "result.embeddings")
  *   .show(5, 80)
  * +---------------+----------+--------------------------------------------------------------------------------+
  * |  annotatorType|    result|                                                                      embeddings|
  * +---------------+----------+--------------------------------------------------------------------------------+
  * |word_embeddings|   This is|[-0.55661, 0.42829502, 0.86661, -0.409785, 0.06316501, 0.120775, -0.0732005, ...|
  * |word_embeddings|      is a|[-0.40674996, 0.22938299, 0.50597, -0.288195, 0.555655, 0.465145, 0.140118, 0...|
  * |word_embeddings|a sentence|[0.17417, 0.095253006, -0.0530925, -0.218465, 0.714395, 0.79860497, 0.0129999...|
  * |word_embeddings|sentence .|[0.139705, 0.177955, 0.1887775, -0.45545, 0.20030999, 0.461557, -0.07891501, ...|
  * +---------------+----------+--------------------------------------------------------------------------------+
  * }}}
  *
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
class ChunkEmbeddings(override val uid: String)
    extends AnnotatorModel[ChunkEmbeddings]
    with HasSimpleAnnotate[ChunkEmbeddings] {

  import com.johnsnowlabs.nlp.AnnotatorType._

  /** Output annotator type : WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = WORD_EMBEDDINGS

  /** Input annotator type : CHUNK, WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(CHUNK, WORD_EMBEDDINGS)

  /** Choose how you would like to aggregate Word Embeddings to Chunk Embeddings: `"AVERAGE"` or
    * `"SUM"` (Default: `"AVERAGE"`)
    *
    * @group param
    */
  val poolingStrategy = new Param[String](
    this,
    "poolingStrategy",
    "Choose how you would like to aggregate Word Embeddings to Chunk Embeddings: AVERAGE or SUM")

  /** Whether to discard default vectors for OOV words from the aggregation / pooling (Default:
    * `true`)
    *
    * @group param
    */
  val skipOOV = new BooleanParam(
    this,
    "skipOOV",
    "Whether to discard default vectors for OOV words from the aggregation / pooling")

  /** PoolingStrategy must be either AVERAGE or SUM
    *
    * @group setParam
    */
  def setPoolingStrategy(strategy: String): this.type = {
    strategy.toLowerCase() match {
      case "average" => set(poolingStrategy, "AVERAGE")
      case "sum" => set(poolingStrategy, "SUM")
      case _ => throw new MatchError("poolingStrategy must be either AVERAGE or SUM")
    }
  }

  /** Whether to discard default vectors for OOV words from the aggregation / pooling
    *
    * @group setParam
    */
  def setSkipOOV(value: Boolean): this.type = set(skipOOV, value)

  /** Choose how you would like to aggregate Word Embeddings to Chunk Embeddings: AVERAGE or SUM
    *
    * @group getParam
    */
  def getPoolingStrategy: String = $(poolingStrategy)

  /** Whether to discard default vectors for OOV words from the aggregation / pooling
    *
    * @group getParam
    */
  def getSkipOOV: Boolean = $(skipOOV)

  setDefault(
    inputCols -> Array(CHUNK, WORD_EMBEDDINGS),
    outputCol -> "chunk_embeddings",
    poolingStrategy -> "AVERAGE",
    skipOOV -> true)

  /** Internal constructor to submit a random UID */
  def this() = this(Identifiable.randomUID("CHUNK_EMBEDDINGS"))

  private def calculateChunkEmbeddings(matrix: Array[Array[Float]]): Array[Float] = {
    val res = Array.ofDim[Float](matrix(0).length)
    matrix(0).indices.foreach { j =>
      matrix.indices.foreach { i =>
        res(j) += matrix(i)(j)
      }
      if ($(poolingStrategy) == "AVERAGE")
        res(j) /= matrix.length
    }
    res
  }

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val documentsWithChunks = annotations
      .filter(token => token.annotatorType == CHUNK)

    val embeddingsSentences = WordpieceEmbeddingsSentence.unpack(annotations)

    documentsWithChunks.flatMap { chunk =>
      val sentenceIdx = chunk.metadata.getOrElse("sentence", "0").toInt
      val chunkIdx = chunk.metadata.getOrElse("chunk", "0").toInt

      if (sentenceIdx < embeddingsSentences.length) {

        val tokensWithEmbeddings = embeddingsSentences(sentenceIdx).tokens.filter(token =>
          token.begin >= chunk.begin && token.end <= chunk.end)

        val allEmbeddings = tokensWithEmbeddings.flatMap(tokenEmbedding =>
          if (!tokenEmbedding.isOOV || ! $(skipOOV))
            Some(tokenEmbedding.embeddings)
          else
            None)

        val finalEmbeddings =
          if (allEmbeddings.length > 0) allEmbeddings else tokensWithEmbeddings.map(_.embeddings)

        /** When we have more chunks than word embeddings this happens when the embeddings has max
          * sequence restriction like BERT, ALBERT, etc.
          */
        if (finalEmbeddings.isEmpty)
          None
        else
          Some(
            Annotation(
              annotatorType = outputAnnotatorType,
              begin = chunk.begin,
              end = chunk.end,
              result = chunk.result,
              metadata = Map(
                "sentence" -> sentenceIdx.toString,
                "chunk" -> chunkIdx.toString,
                "token" -> chunk.result,
                "pieceId" -> "-1",
                "isWordStart" -> "true"),
              embeddings = calculateChunkEmbeddings(finalEmbeddings)))
      } else {
        None
      }
    }
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    val embeddingsCol = Annotation.getColumnByType(dataset, $(inputCols), WORD_EMBEDDINGS)
    dataset.withColumn(
      getOutputCol,
      dataset.col(getOutputCol).as(getOutputCol, embeddingsCol.metadata))
  }

}

/** This is the companion object of [[ChunkEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object ChunkEmbeddings extends DefaultParamsReadable[ChunkEmbeddings]
