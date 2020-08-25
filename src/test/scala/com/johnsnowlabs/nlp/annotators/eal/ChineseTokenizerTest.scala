package com.johnsnowlabs.nlp.annotators.eal

import java.nio.file.{Files, Paths}

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, SparkAccessor}
import org.apache.spark.ml.Pipeline
import org.scalatest.FlatSpec

import scala.collection.mutable

class ChineseTokenizerTest extends FlatSpec {

  val maxWordLength = 2

  import SparkAccessor.spark.implicits._

  private val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  private val sentence = new SentenceDetector()
    .setInputCols("document")
    .setOutputCol("sentence")

  "A ChineseTokenizer" should "tokenize words" in {

    val testDataSet = Seq("十四不是四十").toDS.toDF("text")
    val expectedResult = Array(
      Annotation(TOKEN, 0, 1, "十四", Map("sentence" -> "0")),
      Annotation(TOKEN, 2, 3, "不是", Map("sentence" -> "0")),
      Annotation(TOKEN, 4, 5, "四十", Map("sentence" -> "0"))
    )
    val chineseTokenizer = new ChineseTokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")
      .setKnowledgeBase("src/test/resources/tokenizer/sample_chinese_doc.txt")
      .setMaxWordLength(2)
      .setMinAggregation(1.2)
      .setMinEntropy(0.4)
      .setWordSegmentMethod("ALL")
    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        chineseTokenizer
      ))

    val tokenizerPipeline = pipeline.fit(testDataSet)
    val tokenizerDataSet = tokenizerPipeline.transform(testDataSet)

    val actualResult = tokenizerDataSet
      .select("token.result", "token.metadata", "token.begin",  "token.end").rdd.flatMap{ row=>
      val resultSeq: Seq[String] = row.get(0).asInstanceOf[mutable.WrappedArray[String]]
      val metadataSeq: Seq[Map[String, String]] = row.get(1).asInstanceOf[mutable.WrappedArray[Map[String, String]]]
      val beginSeq: Seq[Int] = row.get(2).asInstanceOf[mutable.WrappedArray[Int]]
      val endSeq: Seq[Int] = row.get(3).asInstanceOf[mutable.WrappedArray[Int]]
      resultSeq.zipWithIndex.map{ case (token, index) =>
        Annotation(TOKEN, beginSeq(index), endSeq(index), token, metadataSeq(index))
      }
    }.collect()
    expectedResult.zipWithIndex.foreach{ case (annotation, index) =>
      assert(annotation.result == actualResult(index).result)
    }
  }

  it should "use documents from dataset as knowledge base" in {
    val testDataSet = Seq(
      "十四是十四四十是四十，十四不是四十，四十不是十四",
      "十四不是四十",
      "十四是十四四十是四十，十四不是四十，四十不是十四").toDS.toDF("text")
    val chineseTokenizer = new ChineseTokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")
      .setMaxWordLength(2)
      .setMinAggregation(1.2)
      .setMinEntropy(0.4)
      .setWordSegmentMethod("ALL")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        chineseTokenizer
      ))

    val tokenizerPipeline = pipeline.fit(testDataSet)
    val tokenizerDataSet = tokenizerPipeline.transform(testDataSet)

    val actualResult = tokenizerDataSet
      .select("token.result", "token.metadata", "token.begin",  "token.end").rdd.map{ row=>
      val resultSeq: Seq[String] = row.get(0).asInstanceOf[mutable.WrappedArray[String]]
      val metadataSeq: Seq[Map[String, String]] = row.get(1).asInstanceOf[mutable.WrappedArray[Map[String, String]]]
      val beginSeq: Seq[Int] = row.get(2).asInstanceOf[mutable.WrappedArray[Int]]
      val endSeq: Seq[Int] = row.get(3).asInstanceOf[mutable.WrappedArray[Int]]
      resultSeq.zipWithIndex.map{ case (token, index) =>
        Annotation(TOKEN, beginSeq(index), endSeq(index), token, metadataSeq(index))
      }
    }.collect()
    assert(actualResult.length == 3)
  }

  it should "serialize a model" in {
    val testDataSet = Seq("十四不是四十").toDS.toDF("text")
    val chineseTokenizer = new ChineseTokenizer()
      .setInputCols("sentence")
      .setOutputCol("token")
      .setKnowledgeBase("src/test/resources/tokenizer/sample_chinese_doc.txt")
      .setMaxWordLength(2)
      .setMinAggregation(1.2)
      .setMinEntropy(0.4)
      .setWordSegmentMethod("ALL")

    chineseTokenizer.fit(testDataSet).write.overwrite().save("./tmp_chinese_tokenizer")
    assertResult(true){
      Files.exists(Paths.get("./tmp_chinese_tokenizer"))
    }
  }

  it should "deserialize a model" in {
    val testDataSet = Seq("十四不是四十").toDS.toDF("text")
    val expectedResult = Array(
      Annotation(TOKEN, 0, 1, "十四", Map("sentence" -> "0")),
      Annotation(TOKEN, 2, 3, "不是", Map("sentence" -> "0")),
      Annotation(TOKEN, 4, 5, "四十", Map("sentence" -> "0"))
    )

    val chineseTokenizer = ChineseTokenizerModel.load("./tmp_chinese_tokenizer")

    val pipeline = new Pipeline()
      .setStages(Array(
        documentAssembler,
        sentence,
        chineseTokenizer
      ))
    val tokenizerPipeline = pipeline.fit(testDataSet)
    val tokenizerDataSet = tokenizerPipeline.transform(testDataSet)
    val actualResult = tokenizerDataSet
      .select("token.result", "token.metadata", "token.begin",  "token.end").rdd.flatMap{ row=>
      val resultSeq: Seq[String] = row.get(0).asInstanceOf[mutable.WrappedArray[String]]
      val metadataSeq: Seq[Map[String, String]] = row.get(1).asInstanceOf[mutable.WrappedArray[Map[String, String]]]
      val beginSeq: Seq[Int] = row.get(2).asInstanceOf[mutable.WrappedArray[Int]]
      val endSeq: Seq[Int] = row.get(3).asInstanceOf[mutable.WrappedArray[Int]]
      resultSeq.zipWithIndex.map{ case (token, index) =>
        Annotation(TOKEN, beginSeq(index), endSeq(index), token, metadataSeq(index))
      }
    }.collect()
    expectedResult.zipWithIndex.foreach{ case (annotation, index) =>
      assert(annotation.result == actualResult(index).result)
    }
  }

}
