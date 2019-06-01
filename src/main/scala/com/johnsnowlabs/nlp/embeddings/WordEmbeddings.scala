package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, TOKEN, WORD_EMBEDDINGS}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.param.{IntParam, Param}
import org.apache.spark.ml.util.{DefaultParamsReadable, Identifiable}
import org.apache.spark.sql.{Dataset, SparkSession}

class WordEmbeddings(override val uid: String) extends AnnotatorApproach[WordEmbeddingsModel] with HasWordEmbeddings {

  def this() = this(Identifiable.randomUID("WORD_EMBEDDINGS"))

  override val outputAnnotatorType: AnnotatorType = WORD_EMBEDDINGS
  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)

  val sourceEmbeddingsPath = new Param[String](this, "sourceEmbeddingsPath", "Word embeddings file")

  val embeddingsFormat = new IntParam(this, "embeddingsFormat", "Word vectors file format")

  override val description: String = "Word Embeddings lookup annotator that maps tokens to vectors"

  def setEmbeddingsSource(path: String, nDims: Int, format: WordEmbeddingsFormat.Format): this.type = {
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, format.id)
    set(this.dimension, nDims)
  }

  def setEmbeddingsSource(path: String, nDims: Int, format: String): this.type = {
    import WordEmbeddingsFormat._
    set(this.sourceEmbeddingsPath, path)
    set(this.embeddingsFormat, format.id)
    set(this.dimension, nDims)
  }

  def setSourcePath(path: String): this.type = set(sourceEmbeddingsPath, path)
  def getSourcePath: String = $(sourceEmbeddingsPath)

  def setEmbeddingsFormat(format: String): this.type = {
    import WordEmbeddingsFormat._
    set(embeddingsFormat, format.id)
  }

  def getEmbeddingsFormat: String = {
    import WordEmbeddingsFormat._
    int2frm($(embeddingsFormat)).toString
  }


  override def beforeTraining(sparkSession: SparkSession): Unit = {
    if (isDefined(sourceEmbeddingsPath)) {
      if (!isLoaded()) {
        EmbeddingsHelper.load(
          $(sourceEmbeddingsPath),
          sparkSession,
          WordEmbeddingsFormat($(embeddingsFormat)).toString,
          $(dimension),
          $(caseSensitive),
          $(embeddingsRef)
        )
        setAsLoaded()
      }
    } else if (isSet(embeddingsRef)) {
      getClusterEmbeddings
    } else
      throw new IllegalArgumentException(
        s"Word embeddings not found. Either sourceEmbeddingsPath not set," +
          s" or not in cache by ref: ${get(embeddingsRef).getOrElse("-embeddingsRef not set-")}. " +
          s"Load using EmbeddingsHelper .loadEmbeddings() and .setEmbeddingsRef() to make them available."
      )
  }

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): WordEmbeddingsModel = {
    val model = new WordEmbeddingsModel()
      .setInputCols($(inputCols))
      .setEmbeddingsRef($(embeddingsRef))
      .setDimension($(dimension))
      .setCaseSensitive($(caseSensitive))
      .setIncludeEmbeddings($(includeEmbeddings))

    getClusterEmbeddings.getLocalRetriever.close()

    model
  }

}

object WordEmbeddings extends DefaultParamsReadable[WordEmbeddings]