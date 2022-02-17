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

package com.johnsnowlabs.ml.tensorflow

import com.johnsnowlabs.nlp.Annotation
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions.{col, explode, size}

import scala.collection.mutable

@SerialVersionUID(112462048007662L)
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

  def encodeTagsMultiLabel(labels: Array[Array[String]]): Array[Array[Float]] = {
    labels.map { t =>
      val labelIDsArray = Array.fill(tags.length)(0.0f)
      if(t.length > 0)
        t.foreach(x=>
          labelIDsArray(tags2Id(x)) = 1.0f
        )
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
    val results = dataset
      .select("embeddings", labelCol)
      .collect()
      .map { row =>
        val newRow = row.get(0).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Float]]].map(x => x.toArray)
        val label = Array(row.getString(1))
        val labelEmbed = newRow.flatMap{e=> Map(label.mkString -> e)}.toArray
        labelEmbed
      }
    results
  }

  /**
    * Converts DataFrame to labels and embeddings
    *
    * @param dataset Input DataFrame with embeddings and labels
    * @return Array of Array of Map(Array(String), Array(Float))
    */
  def collectTrainingInstancesMultiLabel(dataset: DataFrame, labelCol: String): Array[Array[(Array[String], Array[Float])]] = {
    val results = dataset
      .select("embeddings", labelCol)
      .filter(size(col("embeddings")(0)) > 0)
      .rdd
      .map { row =>
        val newRow = row.get(0).asInstanceOf[mutable.WrappedArray[mutable.WrappedArray[Float]]].map(x => x.toArray)
        val label = row.get(1).asInstanceOf[mutable.WrappedArray[String]].toArray
        val labelEmbed = newRow.flatMap{e=> Map(label -> e)}.toArray
        labelEmbed
      }

    System.gc()
    results.collect()
  }

  /**
    * Converts DataFrame to Array of Arrays of Embeddings
    *
    * @param dataset Input DataFrame with sentence_embeddings
    * @return Array of Array of Float
    */
  def extractSentenceEmbeddings(dataset: Array[Array[(String, Array[Float])]]): Array[Array[Float]] = {
    dataset.flatMap{x => x.map{x=>
      val padding = Array.fill[Float](1024 - x._2.length)(0f)
      x._2 ++ padding
    }}
  }

  /**
    * Converts DataFrame to Array of Arrays of Embeddings
    *
    * @param docs Input DataFrame with sentence_embeddings
    * @return Array of Array of Float
    */
  def extractSentenceEmbeddings(docs: Seq[(Int, Seq[Annotation])]): Array[Array[Float]] = {
    docs.map{x =>
      val padding = Array.fill[Float](1024 - x._2.head.embeddings.length)(0f)
      x._2.head.embeddings ++ padding
    }.toArray

  }

  /**
    * Converts DataFrame to Array of arrays of arrays of arrays of Embeddings
    * The difference in this function is to create a sequence in case of multiple sentences in a document
    * Used in MultiClassifierDL
    *
    * @param dataset Input DataFrame with sentence_embeddings
    * @return Array of Arrays of Arrays of Floats
    */
  def extractSentenceEmbeddingsMultiLabel(dataset: Array[Array[(Array[String], Array[Float])]]): Array[Array[Array[Float]]] = {
    dataset.flatMap{x => x.groupBy(x=>x._1).map{ x =>
      x._2.map{y=>
        val padding = Array.fill[Float](1024 - y._2.length)(0f)
        y._2 ++ padding
      }
    }}
  }

  /**
    * Converts DataFrame to Array of arrays of arrays of arrays of Embeddings
    * The difference in this function is to create a sequence in case of multiple sentences in a document
    * Used in MultiClassifierDL
    *
    * @param docs Input DataFrame with sentence_embeddings
    * @return Array of Arrays of Arrays of Floats
    */
  def extractSentenceEmbeddingsMultiLabel(docs: Seq[(Int, Seq[Annotation])]): Array[Array[Array[Float]]] = {
    docs.groupBy(x=>x._1).map{ x =>
      x._2.map(x=>x._2.head.embeddings).toArray
    }.toArray
  }

  def extractSentenceEmbeddingsMultiLabelPredict(docs: Seq[(Int, Seq[Annotation])]): Array[Array[Array[Float]]] = {
    Array(docs.flatMap(x=>x._2.map{x=>
      val padding = Array.fill[Float](1024 - x.embeddings.length)(0f)
      x.embeddings ++ padding
    }).toArray)
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
    * Converts DataFrame to Array of Arrays of Labels (string)
    *
    * @param dataset Input DataFrame with labels
    * @return Array of Array of String
    */
  def extractLabelsMultiLabel(dataset: Array[Array[(Array[String], Array[Float])]]): Array[Array[String]] = {
    dataset.flatMap { x =>
      x.groupBy(x => x._1).keys
    }
  }

  def calculateEmbeddingsDim(dataset: DataFrame): Int = {
    val embedSize = dataset.select(explode(col("embeddings")).as("embedding"))
      .select(size(col("embedding")).as("embeddings_size")).rdd.take(1)
      .map(
        r => r.getInt(0)
      )
    embedSize(0)
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
