package com.johnsnowlabs.ml.logreg

import java.io.File

import com.johnsnowlabs.nlp.annotators.assertion.logreg.Windowing
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsIndexer}
import org.apache.spark.sql._

/**
  * Reader for this dataset,
  * https://github.com/mongoose54/negex/blob/master/genConText/rsAnnotations-1-120-random.txt
  */
class NegexDatasetReader(wordEmbeddingsFile: String, wordEmbeddingsNDims: Int) extends Serializable with Windowing{

  var fileDb = wordEmbeddingsFile + ".db"
  WordEmbeddingsIndexer.indexBinary(wordEmbeddingsFile, fileDb)

  //private val mappings = Map("Affirmed" -> 0.0, "Negated" -> 1.0,"Historical" -> 2.0, "Family" -> 3.0)

  override val (before, after) = (5, 8)

  /* TODO duplicated logic, consider relocation to common place */
  override lazy val wordVectors : Option[WordEmbeddings] =
    if (wordEmbeddingsFile != null) {
    require(new File(wordEmbeddingsFile).exists())
    val fileDb = wordEmbeddingsFile + ".db"
    if (!new File(fileDb).exists())
      WordEmbeddingsIndexer.indexBinary(wordEmbeddingsFile, fileDb)

    if (new File(fileDb).exists())
      Some(WordEmbeddings(fileDb, wordEmbeddingsNDims))
      else
      None
  }
    else
      None

  def readNegexDataset(datasetPath: String)(implicit session:SparkSession) = {
    import session.implicits._

    val dataset = session.read.format("com.databricks.spark.csv").
      option("delimiter", "\t").
      option("header", "true").
      load(datasetPath)

    /* apply UDF to fix the length of each document */
    dataset.select(applyWindowUdf($"sentence", $"target")
      .as("features"), labelToNumber()($"label").as("label"))
  }

  override val embeddingsPath: String = wordEmbeddingsFile
}
