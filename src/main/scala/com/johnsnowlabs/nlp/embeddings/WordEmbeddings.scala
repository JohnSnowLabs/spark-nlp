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

class WordEmbeddings(override val uid: String)
  extends AnnotatorApproach[WordEmbeddingsModel]
    with HasStorage
    with HasEmbeddingsProperties {

  def this() = this(Identifiable.randomUID("WORD_EMBEDDINGS"))

  override val outputAnnotatorType: AnnotatorType = WORD_EMBEDDINGS
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  override val description: String = "Word Embeddings lookup annotator that maps tokens to vectors"

  override protected val missingRefMsg: String = s"Please set storageRef param in $this. This ref is useful for other annotators" +
    " to require this particular set of embeddings. You can use any memorable name such as 'glove' or 'my_embeddings'."

  val writeBufferSize = new IntParam(this, "writeBufferSize", "buffer size limit before dumping to disk storage while writing")
  setDefault(writeBufferSize, 10000)
  def setWriteBufferSize(value: Int): this.type = set(writeBufferSize, value)

  val readCacheSize = new IntParam(this, "readCacheSize", "cache size for items retrieved from storage. Increase for performance but higher memory consumption")
  def setReadCacheSize(value: Int): this.type = set(readCacheSize, value)

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): WordEmbeddingsModel = {
    val model = new WordEmbeddingsModel()
      .setInputCols($(inputCols))
      .setStorageRef($(storageRef))
      .setDimension($(dimension))
      .setCaseSensitive($(caseSensitive))

    if (isSet(readCacheSize))
      model.setReadCacheSize($(readCacheSize))

    model
  }

  override protected def index(
                                fitDataset: Dataset[_],
                                storageSourcePath: Option[String],
                                readAs: Option[ReadAs.Value],
                                writers: Map[Database.Name, StorageWriter[_]],
                                readOptions: Option[Map[String, String]]
                              ): Unit = {
    val writer = writers.values.headOption
      .getOrElse(throw new IllegalArgumentException("Received empty WordEmbeddingsWriter from locators"))
      .asInstanceOf[WordEmbeddingsWriter]

    if (readAs.get == ReadAs.TEXT) {
      WordEmbeddingsTextIndexer.index(storageSourcePath.get, writer)
    }
    else if (readAs.get == ReadAs.BINARY) {
      WordEmbeddingsBinaryIndexer.index(storageSourcePath.get, writer)
    }
    else
      throw new IllegalArgumentException("Invalid WordEmbeddings read format. Must be either TEXT or BINARY")
  }

  override val databases: Array[Database.Name] = WordEmbeddingsModel.databases

  override protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_] = {
    new WordEmbeddingsWriter(connection, $(caseSensitive), $(dimension), get(readCacheSize).getOrElse(5000), $(writeBufferSize))
  }
}

object WordEmbeddings extends DefaultParamsReadable[WordEmbeddings]
