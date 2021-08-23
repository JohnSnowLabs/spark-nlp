/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.collections.StorageSearchTrie
import com.johnsnowlabs.nlp.AnnotatorType.{TOKEN, DOCUMENT, CHUNK}
import com.johnsnowlabs.nlp.annotators.TokenizerModel
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.storage.{Database, HasStorage, RocksDBConnection, StorageWriter}

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.Dataset

/**
 * Annotator to match exact phrases (by token) provided in a file against a Document.
 *
 * A text file of predefined phrases must be provided with `setStoragePath`.
 * The text file can als be set directly as an
 * [[com.johnsnowlabs.nlp.util.io.ExternalResource ExternalResource]].
 *
 * In contrast to the normal `TextMatcher`, the `BigTextMatcher` is designed for large corpora.
 *
 * For extended examples of usage, see the [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/btm/BigTextMatcherTestSpec.scala BigTextMatcherTestSpec]].
 *
 * ==Example==
 * In this example, the entities file is of the form
 * {{{
 * ...
 * dolore magna aliqua
 * lorem ipsum dolor. sit
 * laborum
 * ...
 * }}}
 * where each line represents an entity phrase to be extracted.
 * {{{
 * import spark.implicits._
 * import com.johnsnowlabs.nlp.DocumentAssembler
 * import com.johnsnowlabs.nlp.annotator.Tokenizer
 * import com.johnsnowlabs.nlp.annotator.BigTextMatcher
 * import com.johnsnowlabs.nlp.util.io.ReadAs
 * import org.apache.spark.ml.Pipeline
 *
 * val documentAssembler = new DocumentAssembler()
 *   .setInputCol("text")
 *   .setOutputCol("document")
 *
 * val tokenizer = new Tokenizer()
 *   .setInputCols("document")
 *   .setOutputCol("token")
 *
 * val data = Seq("Hello dolore magna aliqua. Lorem ipsum dolor. sit in laborum").toDF("text")
 * val entityExtractor = new BigTextMatcher()
 *   .setInputCols("document", "token")
 *   .setStoragePath("src/test/resources/entity-extractor/test-phrases.txt", ReadAs.TEXT)
 *   .setOutputCol("entity")
 *   .setCaseSensitive(false)
 *
 * val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, entityExtractor))
 * val results = pipeline.fit(data).transform(data)
 * results.selectExpr("explode(entity)").show(false)
 * +--------------------------------------------------------------------+
 * |col                                                                 |
 * +--------------------------------------------------------------------+
 * |[chunk, 6, 24, dolore magna aliqua, [sentence -> 0, chunk -> 0], []]|
 * |[chunk, 53, 59, laborum, [sentence -> 0, chunk -> 1], []]           |
 * +--------------------------------------------------------------------+
 * }}}
 *
 * @param uid internal uid required to generate writable annotators
 * @groupname anno Annotator types
 * @groupdesc anno Required input and expected output annotator types
 * @groupname Ungrouped Members
 * @groupname param Parameters
 * @groupname setParam Parameter setters
 * @groupname getParam Parameter getters
 * @groupname Ungrouped Members
 * @groupprio anno  1
 * @groupprio param  2
 * @groupprio setParam  3
 * @groupprio getParam  4
 * @groupprio Ungrouped 5
 * @groupdesc param A list of (hyper-)parameter keys this annotator can take. Users can set and get the parameter values through setters and getters, respectively.
 * */
class BigTextMatcher(override val uid: String) extends AnnotatorApproach[BigTextMatcherModel] with HasStorage {

  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  /** Input annotator Types: DOCUMENT, TOKEN
   * @group anno
   */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  /** Output annotator Types: CHUNK
   * @group anno
   */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  override val description: String = "Extracts entities from target dataset given in a text file"

  /** Whether to merge overlapping matched chunks (Default: `false`)
   *
   * @group param
   * */
  val mergeOverlapping = new BooleanParam(this, "mergeOverlapping", "whether to merge overlapping matched chunks. Defaults false")

  /** The Tokenizer to perform tokenization with
   *
   * @group param
   * */
  val tokenizer = new StructFeature[TokenizerModel](this, "tokenizer")

  setDefault(inputCols,Array(TOKEN))
  setDefault(caseSensitive, true)
  setDefault(mergeOverlapping, false)

  /** @group setParam */
  def setTokenizer(tokenizer: TokenizerModel): this.type = set(this.tokenizer, tokenizer)

  /** @group getParam */
  def getTokenizer: TokenizerModel = $$(tokenizer)

  /** @group setParam */
  def setMergeOverlapping(v: Boolean): this.type = set(mergeOverlapping, v)

  /** @group getParam */
  def getMergeOverlapping: Boolean = $(mergeOverlapping)

  /**
    * Loads entities from a provided source.
    */
  private def loadEntities(path: String, writers: Map[Database.Name, StorageWriter[_]]): Unit = {
    val inputFiles: Seq[Iterator[String]] =
      ResourceHelper.parseLinesIterator(ExternalResource(path, ReadAs.TEXT, Map()))
    inputFiles.foreach { inputFile => {
      StorageSearchTrie.load(inputFile, writers, get(tokenizer))
    }}
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): BigTextMatcherModel = {
    new BigTextMatcherModel()
      .setInputCols($(inputCols))
      .setOutputCol($(outputCol))
      .setCaseSensitive($(caseSensitive))
      .setStorageRef($(storageRef))
      .setMergeOverlapping($(mergeOverlapping))
  }

  override protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_] = {
    database match {
      case Database.TMVOCAB => new TMVocabReadWriter(connection, $(caseSensitive))
      case Database.TMEDGES => new TMEdgesReadWriter(connection, $(caseSensitive))
      case Database.TMNODES => new TMNodesWriter(connection)
    }
  }

  override protected def index(
                                fitDataset: Dataset[_],
                                storageSourcePath: Option[String],
                                readAs: Option[ReadAs.Value],
                                writers: Map[Database.Name, StorageWriter[_]],
                                readOptions: Option[Map[String, String]]
                              ): Unit = {
    require(readAs.get == ReadAs.TEXT, "BigTextMatcher only supports TEXT input formats at the moment.")
    loadEntities(storageSourcePath.get, writers)
  }

  override protected val databases: Array[Name] = BigTextMatcherModel.databases
}

/**
 * This is the companion object of [[BigTextMatcher]]. Please refer to that class for the documentation.
 */
object BigTextMatcher extends DefaultParamsReadable[BigTextMatcher]