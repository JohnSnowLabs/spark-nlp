package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.util.io.ReadAs
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.storage.{Database, HasStorage, RocksDBConnection, StorageWriter}
import org.apache.spark.ml.PipelineModel
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

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): WordEmbeddingsModel = {
    val model = new WordEmbeddingsModel()
      .setInputCols($(inputCols))
      .setStorageRef($(storageRef))
      .setDimension($(dimension))
      .setCaseSensitive($(caseSensitive))

    model
  }

  override protected def index(
                                storageSourcePath: String,
                                readAs: ReadAs.Value,
                                writers: Map[Database.Name, StorageWriter[_]],
                                readOptions: Map[String, String]
                              ): Unit = {
    val writer = writers.values.headOption
      .getOrElse(throw new IllegalArgumentException("Received empty WordEmbeddingsWriter from locators"))
      .asInstanceOf[WordEmbeddingsWriter]

    if (readAs == ReadAs.TEXT) {
      WordEmbeddingsTextIndexer.index(storageSourcePath, writer, (1000000.0/$(dimension)).toInt)
    }
    else if (readAs == ReadAs.BINARY) {
      WordEmbeddingsBinaryIndexer.index(storageSourcePath, writer, (1000000.0/$(dimension)).toInt)
    }
    else
      throw new IllegalArgumentException("Invalid WordEmbeddings read format. Must be either TEXT or BINARY")
  }

  override val databases: Array[Database.Name] = Array(Database.EMBEDDINGS)

  override protected def createWriter(database: Name, connection: RocksDBConnection): StorageWriter[_] = {
    new WordEmbeddingsWriter(connection, $(caseSensitive), $(dimension), 5000)
  }
}

object WordEmbeddings extends DefaultParamsReadable[WordEmbeddings]