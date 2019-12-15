package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.nlp.util.io.{ExternalResource, ReadAs}
import com.johnsnowlabs.storage.{HasStorage, RocksDBConnection}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{Dataset, SparkSession}

class WordEmbeddings(override val uid: String)
  extends AnnotatorApproach[WordEmbeddingsModel]
    with HasStorage
    with HasEmbeddingsProperties {

  def this() = this(Identifiable.randomUID("WORD_EMBEDDINGS"))

  override val outputAnnotatorType: AnnotatorType = WORD_EMBEDDINGS
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  override val description: String = "Word Embeddings lookup annotator that maps tokens to vectors"

  override def beforeTraining(spark: SparkSession): Unit = {
    indexStorage(spark, $(storagePath))
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): WordEmbeddingsModel = {
    val model = new WordEmbeddingsModel()
      .setInputCols($(inputCols))
      .setStorageRef($(storageRef))
      .setDimension($(dimension))
      .setCaseSensitive($(caseSensitive))
      .setIncludeStorage($(includeStorage))

    model
  }

  override protected def index(storageSourcePath: String, connection: RocksDBConnection, resource: ExternalResource): Unit = {
    if (resource.readAs == ReadAs.TEXT) {
      WordEmbeddingsTextIndexer.index(storageSourcePath, connection)
    }
    else if (resource.readAs == ReadAs.BINARY) {
      WordEmbeddingsBinaryIndexer.index(storageSourcePath, connection)
    }
    else
      throw new IllegalArgumentException("Invalid WordEmbeddings read format. Must be either TEXT or BINARY")
  }

  override val databases: Array[String] = Array("embeddings")
}

object WordEmbeddings extends DefaultParamsReadable[WordEmbeddings]