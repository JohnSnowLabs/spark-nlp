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

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, SENTENCE_EMBEDDINGS, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.annotators.common.{SentenceSplit, WordpieceEmbeddingsSentence}
import com.johnsnowlabs.nlp.{Annotation, AnnotatorModel, AnnotatorType, HasSimpleAnnotate}
import com.johnsnowlabs.storage.HasStorageRef
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{DataFrame, Dataset}

/** Converts the results from [[WordEmbeddings]], [[BertEmbeddings]], or [[ElmoEmbeddings]] into
  * sentence or document embeddings by either summing up or averaging all the word embeddings in a
  * sentence or a document (depending on the inputCols).
  *
  * This can be configured with `setPoolingStrategy`, which either be `"AVERAGE"` or `"SUM"`.
  *
  * For more extended examples see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/example/python/annotation/text/english/text-similarity/Spark_NLP_Spark_ML_Text_Similarity.ipynb Examples]].
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/SentenceEmbeddingsTestSpec.scala SentenceEmbeddingsTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
  * import com.johnsnowlabs.nlp.embeddings.SentenceEmbeddings
  * import com.johnsnowlabs.nlp.EmbeddingsFinisher
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("document")
  *
  * val tokenizer = new Tokenizer()
  *   .setInputCols(Array("document"))
  *   .setOutputCol("token")
  *
  * val embeddings = WordEmbeddingsModel.pretrained()
  *   .setInputCols("document", "token")
  *   .setOutputCol("embeddings")
  *
  * val embeddingsSentence = new SentenceEmbeddings()
  *   .setInputCols(Array("document", "embeddings"))
  *   .setOutputCol("sentence_embeddings")
  *   .setPoolingStrategy("AVERAGE")
  *
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("sentence_embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *   .setCleanAnnotations(false)
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     tokenizer,
  *     embeddings,
  *     embeddingsSentence,
  *     embeddingsFinisher
  *   ))
  *
  * val data = Seq("This is a sentence.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(5, 80)
  * +--------------------------------------------------------------------------------+
  * |                                                                          result|
  * +--------------------------------------------------------------------------------+
  * |[-0.22093398869037628,0.25130119919776917,0.41810303926467896,-0.380883991718...|
  * +--------------------------------------------------------------------------------+
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
class SentenceEmbeddings(override val uid: String)
    extends AnnotatorModel[SentenceEmbeddings]
    with HasSimpleAnnotate[SentenceEmbeddings]
    with HasEmbeddingsProperties
    with HasStorageRef {

  /** Output annotator type : SENTENCE_EMBEDDINGS
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = SENTENCE_EMBEDDINGS

  /** Input annotator type : DOCUMENT, WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, WORD_EMBEDDINGS)

  /** Number of embedding dimensions (Default: `100`)
    *
    * @group param
    */
  override val dimension = new IntParam(this, "dimension", "Number of embedding dimensions")

  /** Number of embedding dimensions (Default: `100`)
    *
    * @group getParam
    */
  override def getDimension: Int = $(dimension)

  /** Choose how you would like to aggregate Word Embeddings to Sentence Embeddings (Default:
    * `"AVERAGE"`). Can either be `"AVERAGE"` or `"SUM"`.
    *
    * @group param
    */
  val poolingStrategy = new Param[String](
    this,
    "poolingStrategy",
    "Choose how you would like to aggregate Word Embeddings to Sentence Embeddings: AVERAGE or SUM")

  /** Choose how you would like to aggregate Word Embeddings to Sentence Embeddings (Default:
    * `"AVERAGE"`). Can either be `"AVERAGE"` or `"SUM"`.
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

  setDefault(
    inputCols -> Array(DOCUMENT, WORD_EMBEDDINGS),
    outputCol -> "sentence_embeddings",
    poolingStrategy -> "AVERAGE",
    dimension -> 100)

  /** Internal constructor to submit a random UID */
  def this() = this(Identifiable.randomUID("SENTENCE_EMBEDDINGS"))

  private def calculateSentenceEmbeddings(matrix: Array[Array[Float]]): Array[Float] = {
    val res = Array.ofDim[Float](matrix(0).length)
    setDimension(matrix(0).length)

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
    val sentences = SentenceSplit.unpack(annotations)

    val embeddingsSentences = WordpieceEmbeddingsSentence.unpack(annotations)

    sentences.map { sentence =>
      val embeddings =
        embeddingsSentences.filter(embeddings => embeddings.sentenceId == sentence.index)

      val sentenceEmbeddings = embeddings.flatMap { tokenEmbedding =>
        val allEmbeddings = tokenEmbedding.tokens.map { token =>
          token.embeddings
        }
        calculateSentenceEmbeddings(allEmbeddings)
      }.toArray

      Annotation(
        annotatorType = outputAnnotatorType,
        begin = sentence.start,
        end = sentence.end,
        result = sentence.content,
        metadata = Map(
          "sentence" -> sentence.index.toString,
          "token" -> sentence.content,
          "pieceId" -> "-1",
          "isWordStart" -> "true"),
        embeddings = sentenceEmbeddings)
    }
  }
  override protected def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    val ref =
      HasStorageRef.getStorageRefFromInput(dataset, $(inputCols), AnnotatorType.WORD_EMBEDDINGS)
    if (get(storageRef).isEmpty)
      setStorageRef(ref)
    dataset
  }
  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapSentenceEmbeddingsMetadata(
        dataset.col(getOutputCol),
        $(dimension),
        Some($(storageRef))))
  }
}

/** This is the companion object of [[SentenceEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object SentenceEmbeddings extends DefaultParamsReadable[SentenceEmbeddings]
