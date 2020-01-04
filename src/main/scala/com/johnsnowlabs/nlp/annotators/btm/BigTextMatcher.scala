package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.collections.StorageSearchTrie
import com.johnsnowlabs.nlp.AnnotatorType.{TOKEN, _}
import com.johnsnowlabs.nlp.annotators.TokenizerModel
import com.johnsnowlabs.nlp.annotators.param.ExternalResourceParam
import com.johnsnowlabs.nlp.serialization.StructFeature
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs, ResourceHelper}
import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.storage.{Database, HasStorage, RocksDBConnection, StorageWriter}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset

class BigTextMatcher(override val uid: String) extends AnnotatorApproach[BigTextMatcherModel] with HasStorage {

  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  override val inputAnnotatorTypes = Array(DOCUMENT, TOKEN)

  override val outputAnnotatorType: AnnotatorType = CHUNK

  override val description: String = "Extracts entities from target dataset given in a text file"

  val entities = new ExternalResourceParam(this, "entities", "entities external resource.")
  val mergeOverlapping = new BooleanParam(this, "mergeOverlapping", "whether to merge overlapping matched chunks. Defaults false")
  val tokenizer = new StructFeature[TokenizerModel](this, "tokenizer")

  setDefault(inputCols,Array(TOKEN))
  setDefault(caseSensitive, true)
  setDefault(mergeOverlapping, false)

  def setEntities(value: ExternalResource): this.type =
    set(entities, value)

  def setEntities(path: String, readAs: ReadAs.Format, options: Map[String, String] = Map("format" -> "text")): this.type =
    set(entities, ExternalResource(path, readAs, options))

  def setTokenizer(tokenizer: TokenizerModel): this.type = set(this.tokenizer, tokenizer)

  def getTokenizer: TokenizerModel = $$(tokenizer)

  def setMergeOverlapping(v: Boolean): this.type = set(mergeOverlapping, v)

  def getMergeOverlapping: Boolean = $(mergeOverlapping)

  /**
    * Loads entities from a provided source.
    */
  private def loadEntities(path: String, writers: Map[Database.Name, StorageWriter[_]]): Unit = {
    val inputFiles: Seq[Iterator[String]] =
      ResourceHelper.parseLinesIterator(ExternalResource(path, ReadAs.TEXT, Map()))
    inputFiles.foreach { inputFile => {
      StorageSearchTrie.load(inputFile, writers, get(tokenizer), $(caseSensitive))
    }}
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): BigTextMatcherModel = {
    indexStorage(dataset.sparkSession, $(entities))
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
                                storageSourcePath: String,
                                readAs: ReadAs.Value,
                                writers: Map[Database.Name, StorageWriter[_]],
                                readOptions: Map[String, String]
                              ): Unit = {
    require(readAs == ReadAs.TEXT, "BigTextMatcher only supports TEXT input formats at the moment.")
    loadEntities(storageSourcePath, writers)
  }

  override protected val databases: Array[Name] = Array(
    Database.TMVOCAB,
    Database.TMEDGES,
    Database.TMNODES
  )
}

