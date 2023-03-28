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

import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.annotators.common.{
  TokenPieceEmbeddings,
  TokenizedWithSentence,
  WordpieceEmbeddingsSentence
}
import com.johnsnowlabs.nlp.util.io.ResourceHelper.spark.implicits._
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.storage._
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.sql.{DataFrame, Dataset, Row}

/** Word Embeddings lookup annotator that maps tokens to vectors
  *
  * This is the instantiated model of [[WordEmbeddings]].
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val embeddings = WordEmbeddingsModel.pretrained()
  *     .setInputCols("document", "token")
  *     .setOutputCol("embeddings")
  * }}}
  * The default model is `"glove_100d"`, if no name is provided. For available pretrained models
  * please see the [[https://nlp.johnsnowlabs.com/models?task=Embeddings Models Hub]].
  *
  * There are also two convenient functions to retrieve the embeddings coverage with respect to
  * the transformed dataset:
  *   - `withCoverageColumn(dataset, embeddingsCol, outputCol)`: Adds a custom column with word
  *     coverage stats for the embedded field: (`coveredWords`, `totalWords`,
  *     `coveragePercentage`). This creates a new column with statistics for each row.
  *     {{{
  *     val wordsCoverage = WordEmbeddingsModel.withCoverageColumn(resultDF, "embeddings", "cov_embeddings")
  *     wordsCoverage.select("text","cov_embeddings").show(false)
  *     +-------------------+--------------+
  *     |text               |cov_embeddings|
  *     +-------------------+--------------+
  *     |This is a sentence.|[5, 5, 1.0]   |
  *     +-------------------+--------------+
  *     }}}
  *   - `overallCoverage(dataset, embeddingsCol)`: Calculates overall word coverage for the whole
  *     data in the embedded field. This returns a single coverage object considering all rows in
  *     the field.
  *     {{{
  *     val wordsOverallCoverage = WordEmbeddingsModel.overallCoverage(wordsCoverage,"embeddings").percentage
  *     1.0
  *     }}}
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/examples/python/quick_start_offline.ipynb Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddingsTestSpec.scala WordEmbeddingsTestSpec]].
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
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
  * val embeddingsFinisher = new EmbeddingsFinisher()
  *   .setInputCols("embeddings")
  *   .setOutputCols("finished_embeddings")
  *   .setOutputAsVector(true)
  *   .setCleanAnnotations(false)
  *
  * val pipeline = new Pipeline()
  *   .setStages(Array(
  *     documentAssembler,
  *     tokenizer,
  *     embeddings,
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
  * |[-0.570580005645752,0.44183000922203064,0.7010200023651123,-0.417129993438720...|
  * |[-0.542639970779419,0.4147599935531616,1.0321999788284302,-0.4024400115013122...|
  * |[-0.2708599865436554,0.04400600120425224,-0.020260000601410866,-0.17395000159...|
  * |[0.6191999912261963,0.14650000631809235,-0.08592499792575836,-0.2629800140857...|
  * |[-0.3397899866104126,0.20940999686717987,0.46347999572753906,-0.6479200124740...|
  * +--------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[SentenceEmbeddings]] to combine embeddings into a sentence-level representation
  * @see
  *   [[https://nlp.johnsnowlabs.com/docs/en/annotators Annotators Main Page]] for a list of
  *   transformer based embeddings
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
class WordEmbeddingsModel(override val uid: String)
    extends AnnotatorModel[WordEmbeddingsModel]
    with HasSimpleAnnotate[WordEmbeddingsModel]
    with HasEmbeddingsProperties
    with HasStorageModel
    with ParamsAndFeaturesWritable
    with ReadsFromBytes {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("WORD_EMBEDDINGS_MODEL"))

  /** Output annotator type : WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = WORD_EMBEDDINGS

  /** Input annotator type : DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  /** Cache size for items retrieved from storage. Increase for performance but higher memory
    * consumption
    *
    * @group param
    */
  val readCacheSize = new IntParam(
    this,
    "readCacheSize",
    "cache size for items retrieved from storage. Increase for performance but higher memory consumption")

  /** Set cache size for items retrieved from storage. Increase for performance but higher memory
    * consumption
    *
    * @group setParam
    */
  def setReadCacheSize(value: Int): this.type = set(readCacheSize, value)

  private var memoryStorage: Option[Broadcast[Map[BytesKey, Array[Byte]]]] = None

  private def getInMemoryStorage: Map[BytesKey, Array[Byte]] = {
    memoryStorage.map(_.value).getOrElse {
      if ($(enableInMemoryStorage)) {
        getReader(Database.EMBEDDINGS).exportStorageToMap()
      } else {
        Map.empty
      }
    }
  }

  override def beforeAnnotate(dataset: Dataset[_]): Dataset[_] = {
    if (this.memoryStorage.isEmpty && $(enableInMemoryStorage)) {
      val storageReader = getReader(Database.EMBEDDINGS)
      val memoryStorage = storageReader.exportStorageToMap()
      this.memoryStorage = Some(dataset.sparkSession.sparkContext.broadcast(memoryStorage))
    }
    dataset
  }

  /** Takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param annotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val sentences = TokenizedWithSentence.unpack(annotations)
    val withEmbeddings = sentences.map { s =>
      val tokens = s.indexedTokens.map { indexedToken =>
        val (embeddings, zeroArray) = retrieveEmbeddings(indexedToken.token)
        TokenPieceEmbeddings(
          indexedToken.token,
          indexedToken.token,
          -1,
          isWordStart = true,
          embeddings,
          zeroArray,
          indexedToken.begin,
          indexedToken.end)
      }
      WordpieceEmbeddingsSentence(tokens, s.sentenceIndex)
    }

    WordpieceEmbeddingsSentence.pack(withEmbeddings)
  }

  def retrieveEmbeddings(token: String): (Option[Array[Float]], Array[Float]) = {
    if ($(enableInMemoryStorage)) {
      val zeroArray = Array.fill[Float]($(dimension))(0f)
      var embeddings: Option[Array[Float]] = None

      lazy val resultLower =
        getInMemoryStorage.getOrElse(new BytesKey(token.trim.toLowerCase.getBytes()), Array())
      lazy val resultUpper =
        getInMemoryStorage.getOrElse(new BytesKey(token.trim.toUpperCase.getBytes()), Array())
      lazy val resultExact =
        getInMemoryStorage.getOrElse(new BytesKey(token.trim.getBytes()), Array())

      if (resultExact.nonEmpty) {
        embeddings = Some(fromBytes(resultExact))
      } else if (! $(caseSensitive) && resultLower.nonEmpty) {
        embeddings = Some(fromBytes(resultLower))
      } else if (! $(caseSensitive) && resultUpper.nonEmpty) {
        embeddings = Some(fromBytes(resultUpper))
      }

      (embeddings, zeroArray)

    } else {
      val storageReader = getReader(Database.EMBEDDINGS)
      val embeddings = storageReader.lookup(token)
      val zeroArray: Array[Float] = storageReader.emptyValue
      (embeddings, zeroArray)
    }
  }

  override protected def afterAnnotate(dataset: DataFrame): DataFrame = {
    dataset.withColumn(
      getOutputCol,
      wrapEmbeddingsMetadata(dataset.col(getOutputCol), $(dimension), Some($(storageRef))))
  }

  private def bufferSizeFormula: Int = {
    scala.math
      .min( // LRU Cache Size, pick the smallest value up to 50k to reduce memory blue print as dimension grows
        (100.0 / $(dimension)) * 200000,
        50000)
      .toInt
  }

  override protected def createReader(
      database: Database.Name,
      connection: RocksDBConnection): WordEmbeddingsReader = {
    new WordEmbeddingsReader(
      connection,
      $(caseSensitive),
      $(dimension),
      get(readCacheSize).getOrElse(bufferSizeFormula))
  }

  override val databases: Array[Database.Name] = WordEmbeddingsModel.databases
}

trait ReadablePretrainedWordEmbeddings
    extends StorageReadable[WordEmbeddingsModel]
    with HasPretrained[WordEmbeddingsModel] {

  override val databases: Array[Name] = Array(Database.EMBEDDINGS)
  override val defaultModelName: Option[String] = Some("glove_100d")

  /** Java compliant-overrides */
  override def pretrained(): WordEmbeddingsModel = super.pretrained()
  override def pretrained(name: String): WordEmbeddingsModel = super.pretrained(name)
  override def pretrained(name: String, lang: String): WordEmbeddingsModel =
    super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): WordEmbeddingsModel =
    super.pretrained(name, lang, remoteLoc)
}

trait EmbeddingsCoverage {

  case class CoverageResult(covered: Long, total: Long, percentage: Float)

  def withCoverageColumn(
      dataset: DataFrame,
      embeddingsCol: String,
      outputCol: String = "coverage"): DataFrame = {
    val coverageFn = udf((annotatorProperties: Seq[Row]) => {
      val annotations = annotatorProperties.map(Annotation(_))
      val oov =
        annotations.map(x => if (x.metadata.getOrElse("isOOV", "false") == "false") 1 else 0)
      val covered = oov.sum
      val total = annotations.count(_ => true)
      val percentage = 1f * covered / total
      CoverageResult(covered, total, percentage)
    })
    dataset.withColumn(outputCol, coverageFn(col(embeddingsCol)))
  }

  def overallCoverage(dataset: DataFrame, embeddingsCol: String): CoverageResult = {
    val words = dataset
      .select(embeddingsCol)
      .flatMap(row => {
        val annotations = row.getAs[Seq[Row]](embeddingsCol)
        annotations.map(annotation =>
          Tuple2(
            annotation.getAs[Map[String, String]]("metadata")("token"),
            if (annotation
                .getAs[Map[String, String]]("metadata")
                .getOrElse("isOOV", "false") == "false") 1
            else 0))
      })
    val oov = words.reduce((a, b) => Tuple2("Total", a._2 + b._2))
    val covered = oov._2
    val total = words.count()
    val percentage = 1f * covered / total
    CoverageResult(covered, total, percentage)
  }
}

/** This is the companion object of [[WordEmbeddingsModel]]. Please refer to that class for the
  * documentation.
  */
object WordEmbeddingsModel extends ReadablePretrainedWordEmbeddings with EmbeddingsCoverage
