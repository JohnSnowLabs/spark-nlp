package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.annotators.common.TableData
import com.johnsnowlabs.nlp.annotators.sentence_detector_dl.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.base.MultiDocumentAssembler
import com.johnsnowlabs.nlp.{DocumentAssembler, TableAssembler}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

import scala.io.Source

class TapasForQuestionAnsweringTestSpec extends AnyFlatSpec {
  import ResourceHelper.spark.implicits._

  "TapasForQuestionAnswering" should "load saved model" taggedAs SlowTest ignore {
    TapasForQuestionAnswering
      .loadSavedModel("/tmp/tapas_tf", ResourceHelper.spark)
      .setCaseSensitive(false)
      .write
      .overwrite
      .save("/models/sparknlp/tapas")
  }

  "sample" should "yess" in {
    val csvData =
      """
        |"name", "money", "age"
        |"Donald Trump", "$100,000,000", "75"
        |"Elon Musk", "$20,000,000,000,000", "55"
        |""".stripMargin.trim

    val data =Seq(csvData).toDF("json")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("json")
      .setOutputCol("document")

    val tableAssembler = new TableAssembler()
      .setInputCols(Array("document"))
      .setOutputCol("table")
      .setInputFormat("csv")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, tableAssembler)).fit(data)

    val result = pipeline.transform(data)
    result
      .selectExpr("explode(table) AS table")
      .select("table.result", "table.metadata")
      .show(false)
  }
  "TapasForQuestionAnswering" should "convert CSV text to table" in {
    val csv =
      """
        |    Description 1, "Description 2, with comma"
        |    This is a test", ttt
        |      "This is also a test, but with a comma", a
        |    3,
        |    , "aaaa, ""aa"" "
        |    , "test, test"
        |    " ""aa,", "4"" "
        |    ,
        |  1
        |  "sdf sdafsad, asdf"
        |
        |""".stripMargin.trim

    Seq(",", "@", "\t").foreach { delimiter =>
      val csvTable = csv.replace(",", delimiter)
      val tableData =
        TableData.fromJson(TableData.fromCsv(csvTable, delimiter = delimiter).toJson)

      assert(tableData.header.length == 2, s"parsed $delimiter table must have two columns")
      assert(tableData.header.length == 2, s"parsed $delimiter table must have two columns")
      assert(tableData.rows.length == 7, s"parsed $delimiter table must have 6 rows")
    }
  }

  "TapasForQuestionAnswering" should "answer questions" in {
    val sourceFile = Source.fromFile("src/test/resources/tapas/rich_people.json")
    val jsonTableData = sourceFile.getLines().mkString("\n")
    sourceFile.close()

    val questions =
      "Who earns 100,000,000? Who has more money? How much they all earn? How old are they?"
    val data =
      Seq((jsonTableData, questions), (jsonTableData, " "), (jsonTableData, ""), ("", ""))
        .toDF("table_source", "questions")
        .repartition(1)

    val docAssembler = new MultiDocumentAssembler()
      .setInputCols("table_source", "questions")
      .setOutputCols("document_table", "document_questions")

    val sentenceDetector = SentenceDetectorDLModel
      .pretrained()
      .setInputCols(Array("document_questions"))
      .setOutputCol("question")

    val tableAssembler = new TableAssembler()
      .setInputFormat("json")
      .setInputCols(Array("document_table"))
      .setOutputCol("table")

    val tapas = TapasForQuestionAnswering
      .load("/models/sparknlp/tapas")
      .setInputCols(Array("question", "table"))
      .setOutputCol("answer")

    val pipeline =
      new Pipeline().setStages(Array(docAssembler, sentenceDetector, tableAssembler, tapas))
    val pipelineModel = pipeline.fit(data)

    val results1 = pipelineModel
      .transform(data)
      .selectExpr("explode(answer) as answer")
      .selectExpr(
        "answer.metadata.question",
        "answer.result",
        "answer.metadata.cell_positions",
        "answer.metadata.cell_scores")
      .cache()

    results1.show(truncate = false)
    assert(results1.collect().length == 4, "There must be 4 answers")

    // Convert JSON to CSV and run the pipeline again
    val tableData = TableData.fromJson(jsonTableData)
    val csvTableData = (
      tableData.header.map(col => "\"" + col.replace("\"", "\"\"") + "\"").mkString(", ")
        + "\n"
        + tableData.rows
          .map(row => row.map(v => "\"" + v.replace("\"", "\"\"") + "\"").mkString(", "))
          .mkString("\n")
    )

    val csvData = Seq((csvTableData, questions)).toDF("table_source", "questions").repartition(1)
    val tableAssembler2 = new TableAssembler()
      .setInputFormat("csv")
      .setInputCols(Array("document_table"))
      .setOutputCol("table")
    val pipeline2 =
      new Pipeline().setStages(Array(docAssembler, sentenceDetector, tableAssembler2, tapas))
    val pipelineModel2 = pipeline2.fit(data)
    val results2 = pipelineModel2
      .transform(csvData)
      .selectExpr("explode(answer) as answer")
      .selectExpr(
        "answer.metadata.question",
        "answer.result",
        "answer.metadata.cell_positions",
        "answer.metadata.cell_scores")
      .cache()
    results2.show(truncate = false)
    assert(
      results2.collect().length == results1.collect().length,
      "Tapas should return the same number of results for JSON and CSV data")

    results1
      .collect()
      .zip(results2.collect())
      .foreach(x => {
        (0 until x._1.length).map(i => {
          assert(
            x._1.get(i).toString == x._2.get(i).toString,
            "Tapas should return the same results for JSON and CSV data")
        })
      })
  }
}
