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

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import org.apache.spark.sql.Dataset

import scala.collection.mutable

object AssertAnnotations {

  def getActualResult(dataSet: Dataset[_], columnName: String): Array[Seq[Annotation]] = {
    val result = columnName + ".result"
    val metadata = columnName + ".metadata"
    val begin = columnName + ".begin"
    val end = columnName + ".end"
    val embeddings = columnName + ".embeddings"
    dataSet.select(result, metadata, begin,  end, embeddings).rdd.map{ row =>
      val resultDoc = row.getAs[Seq[String]]("result")
      val metadataDoc = row.getAs[Seq[Map[String, String]]]("metadata")
      val beginDoc = row.getAs[Seq[Int]]("begin")
      val endDoc = row.getAs[Seq[Int]]("end")
      val embeddingsDoc = row.getAs[Seq[Seq[Float]]]("embeddings")
      resultDoc.zipWithIndex.map{ case (token, index) =>
        Annotation(TOKEN, beginDoc(index), endDoc(index), token, metadataDoc(index), embeddingsDoc(index).toArray)
      }
    }.collect()
  }

  def assertFields(expectedResult: Array[Seq[Annotation]], actualResult: Array[Seq[Annotation]]): Unit = {
    expectedResult.zipWithIndex.foreach { case (expectedAnnotationDocument, indexDocument) =>
      val actualDocument = actualResult(indexDocument)
      expectedAnnotationDocument.zipWithIndex.foreach { case (expectedAnnotation, index) =>
        val actualResult = actualDocument(index).result
        val actualBegin = actualDocument(index).begin
        val actualEnd = actualDocument(index).end
        val actualMetadata = actualDocument(index).metadata
        val actualEmbeddings = actualDocument(index).embeddings
        val expectedResult = expectedAnnotation.result
        val expectedBegin = expectedAnnotation.begin
        val expectedEnd = expectedAnnotation.end
        val expectedMetadata = expectedAnnotation.metadata
        val expectedEmbeddings = expectedAnnotation.embeddings

        assert(actualResult == expectedResult, s"actual result $actualResult != expected result $expectedResult")
        assert(actualBegin == expectedBegin, s"actual begin $actualBegin != expected begin $expectedBegin")
        assert(actualEnd == expectedEnd, s"actual end $actualEnd != expected end $expectedEnd")
        assert(actualMetadata == expectedMetadata, s"actual metadata $actualMetadata != expected metadata $expectedMetadata")
        assert(actualEmbeddings sameElements expectedEmbeddings, s"actual embeddings ${actualEmbeddings.mkString(" ")} " +
          s"!= expected embeddings ${expectedEmbeddings.mkString(" ")}")
      }
    }
  }

}
