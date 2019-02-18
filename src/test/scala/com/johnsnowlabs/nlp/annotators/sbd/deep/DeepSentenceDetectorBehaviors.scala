package com.johnsnowlabs.nlp.annotators.sbd.deep

import com.johnsnowlabs.nlp.RecursivePipeline
import org.apache.spark.sql.{DataFrame, Dataset}
import org.scalatest.FlatSpec

import scala.collection.mutable

trait DeepSentenceDetectorBehaviors { this: FlatSpec =>

  def transformDataSet(dataSet: Dataset[_], pipeline: RecursivePipeline,
                       expectedResult: Seq[Seq[String]]): Unit = {

    val sentenceDetectorModel = pipeline.fit(dataSet)

    it should "transform a model into a spark dataframe" in {
      val resultDataSet = sentenceDetectorModel.transform(dataSet)
      assert(resultDataSet.isInstanceOf[DataFrame])
    }

    it should "transform to a dataset with segmented sentences" ignore {
      val resultDataSet = sentenceDetectorModel.transform(dataSet)
      val result = getDataFrameAsArray(resultDataSet)

      assert(result == expectedResult)
    }

  }

  def getDataFrameAsArray(dataSet: Dataset[_]): Seq[Seq[String]] = {

    val results = dataSet.select("sentence").collect().flatMap(_.toSeq)
    results.map{result =>
      val resultWrappedArray: mutable.WrappedArray[String] = result.asInstanceOf[mutable.WrappedArray[String]]
      val resultSeq: Seq[String] = resultWrappedArray
      resultSeq
    }

  }

}
