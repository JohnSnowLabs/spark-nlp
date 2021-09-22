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

package com.johnsnowlabs.nlp.annotators.btm

import com.johnsnowlabs.collections.StorageSearchTrie
import com.johnsnowlabs.nlp.AnnotatorType._
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.storage.Database.Name
import com.johnsnowlabs.storage.{Database, HasStorageModel, RocksDBConnection, StorageReadable, StorageReader}

import org.apache.spark.ml.param.BooleanParam
import org.apache.spark.ml.util.Identifiable

import scala.annotation.{tailrec => tco}
import scala.collection.mutable.ArrayBuffer

/**
 *
 * Instantiated model of the [[BigTextMatcher]].
 * For usage and examples see the documentation of the main class.
 *
 * @param uid internally renquired UID to make it writable
 */
class BigTextMatcherModel(override val uid: String) extends AnnotatorModel[BigTextMatcherModel] with HasSimpleAnnotate[BigTextMatcherModel] with HasStorageModel {

  /** Output Annotator Types: CHUNK
   *
   * @group anno
   */
  override val outputAnnotatorType: AnnotatorType = CHUNK

  /** Input Annotator Types: DOCUMENT, TOKEN
   *
   * @group anno
   */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT, TOKEN)

  /** Whether to merge overlapping matched chunks (Default: `false`)
   *
   * @group param
   * */
  val mergeOverlapping = new BooleanParam(this, "mergeOverlapping", "whether to merge overlapping matched chunks. Defaults false")

  /** @group setParam */
  def setMergeOverlapping(v: Boolean): this.type = set(mergeOverlapping, v)

  /** @group getParam */
  def getMergeOverlapping: Boolean = $(mergeOverlapping)

  /** internal constructor for writabale annotator */
  def this() = this(Identifiable.randomUID("ENTITY_EXTRACTOR"))

  @tco final protected def collapse(rs: List[(Int, Int)], sep: List[(Int, Int)] = Nil): List[(Int, Int)] = rs match {
    case x :: y :: rest =>
      if (y._1 > x._2) collapse(y :: rest, x :: sep)
      else collapse((x._1, x._2 max y._2) :: rest, sep)
    case _ =>
      (rs ::: sep).reverse
  }

  protected def merge(rs: List[(Int, Int)]): List[(Int, Int)] = collapse(rs.sortBy(_._1))

  @transient private lazy val searchTrie = new StorageSearchTrie(
    getReader(Database.TMVOCAB).asInstanceOf[TMVocabReader],
    getReader(Database.TMEDGES).asInstanceOf[TMEdgesReader],
    getReader(Database.TMNODES).asInstanceOf[TMNodesReader]
  )

  /**
   * Searches entities and stores them in the annotation
   *
   * @return Extracted Entities
   */

  /** Defines annotator phrase matching depending on whether we are using SBD or not */
  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {

    val result = ArrayBuffer[Annotation]()

    val sentences = annotations.filter(_.annotatorType == AnnotatorType.DOCUMENT)

    sentences.zipWithIndex.foreach { case (sentence, sentenceIndex) =>

      val tokens = annotations.filter(token =>
        token.annotatorType == AnnotatorType.TOKEN &&
          token.begin >= sentence.begin &&
          token.end <= sentence.end)

      val foundTokens = searchTrie.search(tokens.map(_.result)).toList

      val finalTokens = if ($(mergeOverlapping)) merge(foundTokens) else foundTokens

      for ((begin, end) <- finalTokens) {

        val firstTokenBegin = tokens(begin).begin
        val lastTokenEnd = tokens(end).end

        /** token indices are not relative to sentence but to document, adjust offset accordingly */
        val normalizedText = sentence.result.substring(firstTokenBegin - sentence.begin, lastTokenEnd - sentence.begin + 1)

        val annotation = Annotation(
          outputAnnotatorType,
          firstTokenBegin,
          lastTokenEnd,
          normalizedText,
          Map("sentence" -> sentenceIndex.toString, "chunk" -> result.length.toString)
        )

        result.append(annotation)
      }
    }

    result
  }

  override protected val databases: Array[Name] = BigTextMatcherModel.databases

  override protected def createReader(database: Name, connection: RocksDBConnection): StorageReader[_] = {
    database match {
      case Database.TMVOCAB => new TMVocabReader(connection, $(caseSensitive))
      case Database.TMEDGES => new TMEdgesReader(connection, $(caseSensitive))
      case Database.TMNODES => new TMNodesReader(connection, $(caseSensitive))
    }
  }
}

trait ReadablePretrainedBigTextMatcher extends StorageReadable[BigTextMatcherModel] with HasPretrained[BigTextMatcherModel] {
  override val databases: Array[Name] = Array(
    Database.TMVOCAB,
    Database.TMEDGES,
    Database.TMNODES
  )
  override val defaultModelName = None

  override def pretrained(): BigTextMatcherModel = super.pretrained()

  override def pretrained(name: String): BigTextMatcherModel = super.pretrained(name)

  override def pretrained(name: String, lang: String): BigTextMatcherModel = super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): BigTextMatcherModel = super.pretrained(name, lang, remoteLoc)
}

/**
 * This is the companion object of [[BigTextMatcherModel]]. Please refer to that class for the documentation.
 */
object BigTextMatcherModel extends ReadablePretrainedBigTextMatcher