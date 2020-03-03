package com.johnsnowlabs.ml.tensorflow

import org.apache.spark.sql.DataFrame
import scala.collection.mutable

class ClassifierDatasetEncoder(val params: ClassifierDatasetEncoderParams) extends Serializable {

  val tags2Id: Map[String, Int] = params.tags.zipWithIndex
    .map(p => (p._1, p._2))
    .toMap

  val tags: Array[String] = tags2Id
    .map(p => (p._2, p._1))
    .toArray
    .sortBy(p => p._1)
    .map(p => p._2)

  def encodeTags(labels: Array[String]): Array[Array[Int]] = {
    labels.map { t =>
      val labelIDsArray = Array.fill(tags.length)(0)
      labelIDsArray(tags2Id(t)) = 1
      labelIDsArray
    }
  }

  /**
    * Converts DataFrame to Array of Arrays of Labels (string)
    *
    * @param dataset Input DataFrame with embeddings and labels
    * @return Array of Array of Map(String, Array(Float))
    */
  def collectTrainingInstances(dataset: DataFrame, labelCol: String): Array[Array[(String, Array[Float])]] = {
    dataset
      .select("embeddings", labelCol)
      .rdd
      .map { row =>
        val newRow = row.get(0).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Float]]].map(x => x.toArray)
        val label = Array(row.getString(1))

        val labelEmbed = newRow.flatMap{e=>
          Map(label.mkString -> e)
        }.toArray
        labelEmbed
      }
      .collect()
  }

  /**
    * Converts DataFrame to Array of Arrays of Embeddings
    *
    * @param dataset Input DataFrame with sentence_embeddings
    * @return Array of Array of Float
    */
  def extractSentenceEmbeddings(dataset: Array[Array[(String, Array[Float])]]): Array[Array[Float]] = {
    dataset.flatMap{x => x.map(x=>x._2)}
  }

  /**
    * Converts DataFrame to Array of Arrays of Labels (string)
    *
    * @param dataset Input DataFrame with labels
    * @return Array of Array of String
    */
  def extractLabels(dataset: Array[Array[(String, Array[Float])]]): Array[String] = {
    dataset.flatMap{x => x.map(x=>x._1)}
  }

  /**
    * Converts Tag Identifiers to Tag Names
    *
    * @param tagIds Tag Ids encoded for Tensorflow Model.
    * @return Tag names
    */
  def decodeOutputData(tagIds: Array[Array[Float]]): Array[Array[(String, Float)]] = {
    val scoresMetadata = tagIds.map { scores =>
      scores.zipWithIndex.flatMap {
        case (score, idx) =>
          val tag = tags2Id.find(_._2 == idx).map(_._1).getOrElse("NA")
          Map(tag -> score)
      }
    }

    scoresMetadata
  }
}

case class ClassifierDatasetEncoderParams(tags: Array[String])
