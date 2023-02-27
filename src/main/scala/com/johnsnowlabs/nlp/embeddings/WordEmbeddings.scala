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

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.util.io.ReadAs
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.storage.{Database, HasStorage, RocksDBConnection, StorageWriter}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.IntParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/** Word Embeddings lookup annotator that maps tokens to vectors.
  *
  * For instantiated/pretrained models, see [[WordEmbeddingsModel]].
  *
  * A custom token lookup dictionary for embeddings can be set with `setStoragePath`. Each line of
  * the provided file needs to have a token, followed by their vector representation, delimited by
  * a spaces.
  * {{{
  * ...
  * are 0.39658191506190343 0.630968081620067 0.5393722253731201 0.8428180123359783
  * were 0.7535235923631415 0.9699218875629833 0.10397182122983872 0.11833962569383116
  * stress 0.0492683418305907 0.9415954572751959 0.47624463167525755 0.16790967216778263
  * induced 0.1535748762292387 0.33498936903209897 0.9235178224122094 0.1158772920395934
  * ...
  * }}}
  * If a token is not found in the dictionary, then the result will be a zero vector of the same
  * dimension. Statistics about the rate of converted tokens, can be retrieved with
  * [[WordEmbeddingsModel WordEmbeddingsModel.withCoverageColumn]] and
  * [[WordEmbeddingsModel WordEmbeddingsModel.overallCoverage]].
  *
  * For extended examples of usage, see the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/scala/training/NerDL/win/customNerDlPipeline/CustomForNerDLPipeline.java Examples]]
  * and the
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/embeddings/WordEmbeddingsTestSpec.scala WordEmbeddingsTestSpec]].
  *
  * ==Example==
  * In this example, the file `random_embeddings_dim4.txt` has the form of the content above.
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.Tokenizer
  * import com.johnsnowlabs.nlp.embeddings.WordEmbeddings
  * import com.johnsnowlabs.nlp.util.io.ReadAs
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
  * val embeddings = new WordEmbeddings()
  *   .setStoragePath("src/test/resources/random_embeddings_dim4.txt", ReadAs.TEXT)
  *   .setStorageRef("glove_4d")
  *   .setDimension(4)
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
  * val data = Seq("The patient was diagnosed with diabetes.").toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * result.selectExpr("explode(finished_embeddings) as result").show(false)
  * +----------------------------------------------------------------------------------+
  * |result                                                                            |
  * +----------------------------------------------------------------------------------+
  * |[0.9439099431037903,0.4707513153553009,0.806300163269043,0.16176554560661316]     |
  * |[0.7966810464859009,0.5551124811172485,0.8861005902290344,0.28284206986427307]    |
  * |[0.025029370561242104,0.35177749395370483,0.052506182342767715,0.1887107789516449]|
  * |[0.08617766946554184,0.8399239182472229,0.5395117998123169,0.7864698767662048]    |
  * |[0.6599600911140442,0.16109347343444824,0.6041093468666077,0.8913561105728149]    |
  * |[0.5955275893211365,0.01899011991918087,0.4397728443145752,0.8911281824111938]    |
  * |[0.9840458631515503,0.7599489092826843,0.9417727589607239,0.8624503016471863]     |
  * +----------------------------------------------------------------------------------+
  * }}}
  *
  * @see
  *   [[SentenceEmbeddings]] to combine embeddings into a sentence-level representation
  * @see
  *   [[https://nlp.johnsnowlabs.com/docs/en/annotators Annotators Main Page]] for a list of
  *   transformer based embeddings
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
class WordEmbeddings(override val uid: String)
    extends AnnotatorApproach[WordEmbeddingsModel]
    with HasStorage
    with HasEmbeddingsProperties {

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator
    * type
    */
  def this() = this(Identifiable.randomUID("WORD_EMBEDDINGS"))

  /** Output annotation type : WORD_EMBEDDINGS
    *
    * @group anno
    */
  override val outputAnnotatorType: AnnotatorType = WORD_EMBEDDINGS

  /** Input annotation type : DOCUMENT, TOKEN
    *
    * @group anno
    */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  /** Word Embeddings lookup annotator that maps tokens to vectors */
  override val description: String =
    "Word Embeddings lookup annotator that maps tokens to vectors"

  /** Error message */
  override protected val missingRefMsg: String =
    s"Please set storageRef param in $this. This ref is useful for other annotators" +
      " to require this particular set of embeddings. You can use any memorable name such as 'glove' or 'my_embeddings'."

  /** Buffer size limit before dumping to disk storage while writing
    *
    * @group param
    */
  val writeBufferSize = new IntParam(
    this,
    "writeBufferSize",
    "Buffer size limit before dumping to disk storage while writing")
  setDefault(writeBufferSize, 10000)

  /** Buffer size limit before dumping to disk storage while writing.
    *
    * @group setParam
    */
  def setWriteBufferSize(value: Int): this.type = set(writeBufferSize, value)

  /** Cache size for items retrieved from storage. Increase for performance but higher memory
    * consumption
    *
    * @group param
    */
  val readCacheSize = new IntParam(
    this,
    "readCacheSize",
    "Cache size for items retrieved from storage. Increase for performance but higher memory consumption")

  /** Cache size for items retrieved from storage. Increase for performance but higher memory
    * consumption.
    *
    * @group setParam
    */
  def setReadCacheSize(value: Int): this.type = set(readCacheSize, value)

  override def train(
      dataset: Dataset[_],
      recursivePipeline: Option[PipelineModel]): WordEmbeddingsModel = {

    val model = new WordEmbeddingsModel()
      .setInputCols($(inputCols))
      .setStorageRef($(storageRef))
      .setDimension($(dimension))
      .setCaseSensitive($(caseSensitive))
      .setEnableInMemoryStorage($(enableInMemoryStorage))

    if (isSet(readCacheSize))
      model.setReadCacheSize($(readCacheSize))

    model
  }

  override protected def index(
      fitDataset: Dataset[_],
      storageSourcePath: Option[String],
      readAs: Option[ReadAs.Value],
      writers: Map[Database.Name, StorageWriter[_]],
      readOptions: Option[Map[String, String]]): Unit = {
    val writer = writers.values.headOption
      .getOrElse(
        throw new IllegalArgumentException("Received empty WordEmbeddingsWriter from locators"))
      .asInstanceOf[WordEmbeddingsWriter]

    if (readAs.get == ReadAs.TEXT) {
      WordEmbeddingsTextIndexer.index(storageSourcePath.get, writer)
    } else if (readAs.get == ReadAs.BINARY) {
      WordEmbeddingsBinaryIndexer.index(storageSourcePath.get, writer)
    } else
      throw new IllegalArgumentException(
        "Invalid WordEmbeddings read format. Must be either TEXT or BINARY")
  }

  override val databases: Array[Database.Name] = WordEmbeddingsModel.databases

  override protected def createWriter(
      database: Name,
      connection: RocksDBConnection): StorageWriter[_] = {
    new WordEmbeddingsWriter(
      connection,
      $(caseSensitive),
      $(dimension),
      get(readCacheSize).getOrElse(5000),
      $(writeBufferSize))
  }
}

/** This is the companion object of [[WordEmbeddings]]. Please refer to that class for the
  * documentation.
  */
object WordEmbeddings extends DefaultParamsReadable[WordEmbeddings]
