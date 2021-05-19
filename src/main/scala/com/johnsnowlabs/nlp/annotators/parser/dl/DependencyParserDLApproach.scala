/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.parser.dl

import com.johnsnowlabs.nlp.AnnotatorApproach
import com.johnsnowlabs.nlp.AnnotatorType.{DOCUMENT, LABELED_DEPENDENCY, TOKEN}
import com.johnsnowlabs.util.spark.SparkSqlHelper
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.Dataset
import org.apache.spark.sql.functions.col

class DependencyParserDLApproach(override val uid: String) extends AnnotatorApproach[DependencyParserDLModel] {

  def this() = this(Identifiable.randomUID("DEPENDENCY_PARSER_DL"))

  override val description: String = "Dependency Parser DL finds a grammatical relation between two words in a sentence"

  override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): DependencyParserDLModel = {
    //TODO: Check if column embeddings are in the dataset to avoid computing vocabulary
    //TODO: Control that either dataset is sent with CoNLLU format or create a parameter for lemma.result
    //TODO: Add a parameter to take a sample from dataset to compute vocabulary
    val dataSetWithUniqueWords =
      SparkSqlHelper.uniqueArrayElements(dataset.withColumn("words", col("lemma.result")), "words")

    val uniqueWords = Seq("*INITIAL*") ++ dataSetWithUniqueWords.select("unique_words_elements").rdd.map(rows =>
      rows.getSeq(0).asInstanceOf[Seq[String]]).collect().flatten.distinct

    val vocabulary: Map[String, Int] = uniqueWords.zipWithIndex.map( word => (word._1, word._2)).toMap

    new DependencyParserDLModel()
      .setVocabulary(vocabulary)
  }

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(DOCUMENT, TOKEN)
  override val outputAnnotatorType: AnnotatorType = LABELED_DEPENDENCY
}
