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

package com.johnsnowlabs.nlp.annotators.common

import com.johnsnowlabs.ml.crf.TextSentenceLabels
import com.johnsnowlabs.ml.tensorflow.SentenceGrouper
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.catalyst.encoders.RowEncoder

import scala.reflect.ClassTag

object DatasetHelpers {

  implicit class DataFrameHelper(dataset: DataFrame) {
    def randomize: DataFrame = {
      implicit val encoder = RowEncoder(dataset.schema)
      dataset.mapPartitions {
        new scala.util.Random().shuffle(_).toIterator
      }
    }
  }
  
  def doSlice[T: ClassTag](dataset: TraversableOnce[T], getLen: T => Int, batchSize: Int = 32): Iterator[Array[T]] = {
    val gr = SentenceGrouper[T](getLen)
    gr.slice(dataset, batchSize)
  }

  def slice(dataset: TraversableOnce[(TextSentenceLabels, WordpieceEmbeddingsSentence)], batchSize: Int = 32):
  Iterator[Array[(TextSentenceLabels, WordpieceEmbeddingsSentence)]] = {
    doSlice[(TextSentenceLabels, WordpieceEmbeddingsSentence)](dataset, _._2.tokens.length, batchSize)
  }

}
