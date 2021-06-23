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
    dataSet.select(result, metadata, begin,  end).rdd.map{ row=>
      val resultSeq: Seq[String] = row.get(0).asInstanceOf[mutable.WrappedArray[String]]
      val metadataSeq: Seq[Map[String, String]] = row.get(1).asInstanceOf[mutable.WrappedArray[Map[String, String]]]
      val beginSeq: Seq[Int] = row.get(2).asInstanceOf[mutable.WrappedArray[Int]]
      val endSeq: Seq[Int] = row.get(3).asInstanceOf[mutable.WrappedArray[Int]]
      resultSeq.zipWithIndex.map{ case (token, index) =>
        Annotation(TOKEN, beginSeq(index), endSeq(index), token, metadataSeq(index))
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
        val expectedResult = expectedAnnotation.result
        val expectedBegin = expectedAnnotation.begin
        val expectedEnd = expectedAnnotation.end
        val expectedMetadata = expectedAnnotation.metadata
        assert(actualResult == expectedResult, s"actual result $actualResult != expected result $expectedResult")
        assert(actualBegin == expectedBegin, s"actual begin $actualBegin != expected result $expectedBegin")
        assert(actualEnd == expectedEnd, s"actual end $actualEnd != expected end $expectedEnd")
        assert(actualMetadata == expectedMetadata, s"actual begin $actualMetadata != expected result $expectedMetadata")
      }
    }
  }

}
