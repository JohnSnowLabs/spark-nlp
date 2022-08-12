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

package com.johnsnowlabs.nlp

import org.apache.spark.sql.Dataset

import scala.collection.mutable

object AssertAnnotations {

  def getActualResult(dataSet: Dataset[_], columnName: String): Array[Seq[Annotation]] = {
    val result = columnName + ".result"
    val metadata = columnName + ".metadata"
    val begin = columnName + ".begin"
    val end = columnName + ".end"
    val annotatorType = columnName + ".annotatorType"
    val embeddings = columnName + ".embeddings"

    dataSet
      .select(result, metadata, begin, end, annotatorType, embeddings)
      .rdd
      .map { row =>
        val resultSeq: Seq[String] =
          row.getAs[String]("result").asInstanceOf[mutable.WrappedArray[String]]
        val metadataSeq: Seq[Map[String, String]] = row
          .getAs[Map[String, String]]("metadata")
          .asInstanceOf[mutable.WrappedArray[Map[String, String]]]
        val beginSeq: Seq[Int] = row.getAs[Int]("begin").asInstanceOf[mutable.WrappedArray[Int]]
        val endSeq: Seq[Int] = row.getAs[Int]("end").asInstanceOf[mutable.WrappedArray[Int]]
        val annotatorTypeSeq: Seq[String] = row
          .getAs[String]("annotatorType")
          .asInstanceOf[mutable.WrappedArray[String]]
        val embeddings: Seq[Seq[Float]] = row
          .getAs[Seq[Float]]("embeddings")
          .asInstanceOf[mutable.WrappedArray[Seq[Float]]]

        resultSeq.zipWithIndex.map { case (token, index) =>
          Annotation(
            annotatorTypeSeq(index),
            beginSeq(index),
            endSeq(index),
            token,
            metadataSeq(index),
            embeddings(index).toArray)
        }
      }
      .collect()
  }

  def assertFields(
      expectedResult: Array[Seq[Annotation]],
      actualResult: Array[Seq[Annotation]]): Unit = {
    expectedResult.zipWithIndex.foreach { case (expectedAnnotationDocument, indexDocument) =>
      val actualDocument = actualResult(indexDocument)
      expectedAnnotationDocument.zipWithIndex.foreach { case (expectedAnnotation, index) =>
        val actualResult = actualDocument(index).result
        val actualBegin = actualDocument(index).begin
        val actualEnd = actualDocument(index).end
        val actualMetadata = actualDocument(index).metadata
        val actualAnnotatorType = actualDocument(index).annotatorType
        val expectedResult = expectedAnnotation.result
        val expectedBegin = expectedAnnotation.begin
        val expectedEnd = expectedAnnotation.end
        val expectedMetadata = expectedAnnotation.metadata
        val expectedAnnotatorType = expectedAnnotation.annotatorType
        assert(
          actualResult == expectedResult,
          s"actual result $actualResult != expected result $expectedResult")
        assert(
          actualBegin == expectedBegin,
          s"actual begin $actualBegin != expected result $expectedBegin")
        assert(actualEnd == expectedEnd, s"actual end $actualEnd != expected end $expectedEnd")
        assert(
          actualMetadata == expectedMetadata,
          s"actual begin $actualMetadata != expected result $expectedMetadata")
        assert(
          actualAnnotatorType == expectedAnnotatorType,
          s"actual annotatorType $actualMetadata != expected annotatorType $expectedMetadata")
      }
    }
  }

  def getActualImageResult(dataSet: Dataset[_], columnName: String): Array[Seq[AnnotationImage]] = {
    val annotatorType = columnName + ".annotatorType"
    val origin = columnName + ".origin"
    val height = columnName + ".height"
    val width = columnName + ".width"
    val nChannels = columnName + ".nChannels"
    val mode = columnName + ".mode"
    val result = columnName + ".result"
    val metadata = columnName + ".metadata"

    dataSet
      .select(annotatorType, origin, height, width, nChannels, mode, result, metadata)
      .rdd
      .map { row =>
        val annotatorTypeSeq: Seq[String] = row
          .getAs[String]("annotatorType")
          .asInstanceOf[mutable.WrappedArray[String]]
        val originSeq: Seq[String] = row
          .getAs[String]("origin")
          .asInstanceOf[mutable.WrappedArray[String]]
        val heightSeq: Seq[Int] = row.getAs[Int]("height")
          .asInstanceOf[mutable.WrappedArray[Int]]
        val widthSeq: Seq[Int] = row.getAs[Int]("width")
          .asInstanceOf[mutable.WrappedArray[Int]]
        val nChannelsSeq: Seq[Int] = row.getAs[Int]("nChannels")
          .asInstanceOf[mutable.WrappedArray[Int]]
        val modeSeq: Seq[Int] = row.getAs[Int]("mode")
          .asInstanceOf[mutable.WrappedArray[Int]]
        val resultSeq: Seq[Seq[Byte]] = row
          .getAs[Seq[Byte]]("result")
          .asInstanceOf[mutable.WrappedArray[Seq[Byte]]]
        val metadataSeq: Seq[Map[String, String]] = row
          .getAs[Map[String, String]]("metadata")
          .asInstanceOf[mutable.WrappedArray[Map[String, String]]]

        originSeq.zipWithIndex.map { case (origin, index) =>
          AnnotationImage(
            annotatorTypeSeq(index),
            origin,
            heightSeq(index),
            widthSeq(index),
            nChannelsSeq(index),
            modeSeq(index),
            resultSeq(index).asInstanceOf[Array[Byte]],
            metadataSeq(index)
            )
        }
      }
      .collect()
  }

}
