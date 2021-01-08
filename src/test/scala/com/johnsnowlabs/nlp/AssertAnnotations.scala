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
    expectedResult.zipWithIndex.foreach { case (annotationDocument, indexDocument) =>
      val actualDocument = actualResult(indexDocument)
      annotationDocument.zipWithIndex.foreach { case (annotation, index) =>
        assert(actualDocument(index).result == annotation.result)
        assert(actualDocument(index).begin == annotation.begin)
        assert(actualDocument(index).end == annotation.end)
        assert(actualDocument(index).metadata == annotation.metadata)
      }
    }
  }

}
