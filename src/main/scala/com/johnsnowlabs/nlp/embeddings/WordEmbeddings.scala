package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN, WORD_EMBEDDINGS}
import com.johnsnowlabs.storage.{HasStorage, StorageHelper}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{Dataset, SparkSession}

class WordEmbeddings(override val uid: String) extends AnnotatorApproach[WordEmbeddingsModel] with HasStorage[Float] with HasEmbeddingsProperties {

  def this() = this(Identifiable.randomUID("WORD_EMBEDDINGS"))

  override val outputAnnotatorType: AnnotatorType = WORD_EMBEDDINGS
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  val sourceEmbeddingsPath = new Param[String](this, "sourceEmbeddingsPath", "Word embeddings file")

  val embeddingsFormat = new IntParam(this, "embeddingsFormat", "Word vectors file format")

  override val description: String = "Word Embeddings lookup annotator that maps tokens to vectors"

  def setEmbeddingsSource(path: String, nDims: Int, format: EmbeddingsFormat.Format): this.type = {
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, format.id)
  }

  def setEmbeddingsSource(path: String, nDims: Int, format: String): this.type = {
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, EmbeddingsFormat.withName(format.toUpperCase).id)
  }

  def setSourcePath(path: String): this.type = set(sourceEmbeddingsPath, path)
  def getSourcePath: String = $(sourceEmbeddingsPath)

  def setEmbeddingsFormat(format: String): this.type = {
    set(embeddingsFormat, EmbeddingsFormat.withName(format.toUpperCase).id)
  }

  def getEmbeddingsFormat: String = {
    EmbeddingsFormat.apply($(embeddingsFormat)).toString
  }


  override def beforeTraining(sparkSession: SparkSession): Unit = {
    if (isDefined(sourceEmbeddingsPath)) {
      if (!storageIsReady) {
        WordEmbeddingsIndexer.indexStorage(
          $(sourceEmbeddingsPath),
          $(storageRef),
          EmbeddingsFormat.apply($(embeddingsFormat)),
          sparkSession.sparkContext
        )
      }
      setAndGetStorageConnection
    } else if (isSet(storageRef)) {
      setAndGetStorageConnection
    } else
      throw new IllegalArgumentException(
        s"Word embeddings not found. Either sourceEmbeddingsPath not set," +
          s" or not in cache by ref: ${get(storageRef).getOrElse("-storageRef not set-")}. " +
          s"Load using EmbeddingsHelper.load() and .setStorageRef() to make them available."
      )
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): WordEmbeddingsModel = {
    val model = new WordEmbeddingsModel()
      .setInputCols($(inputCols))
      .setStorageRef($(storageRef))
      .setDimension($(dimension))
      .setCaseSensitive($(caseSensitive))
      .setIncludeStorage($(includeStorage))
      .setStorage(setAndGetStorageConnection)

    model
  }

}

object WordEmbeddings extends DefaultParamsReadable[WordEmbeddings]