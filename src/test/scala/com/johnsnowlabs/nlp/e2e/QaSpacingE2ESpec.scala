package com.johnsnowlabs.nlp.e2e

import com.johnsnowlabs.nlp.SparkAccessor.spark
import com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForQuestionAnswering
import com.johnsnowlabs.nlp.base.{DocumentAssembler, MultiDocumentAssembler}
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec

class QaSpacingE2ESpec extends AnyFlatSpec {
  import spark.implicits._

  "RoBertaForQuestionAnswering" should "not leave stray spaces around punctuation/contractions in the answer" taggedAs SlowTest in {
    val data = Seq(
      (
        "What is the name of the stadium?",
        "Levi's Stadium is the home field of the San Francisco 49ers."),
      (
        "What team does he play for?",
        "It's well known that John plays for the Golden State Warriors."),
      (
        "What was the company's slogan?",
        "The company's slogan was \"Don't be evil\" for many years."))
      .toDF("question", "context")
      .repartition(1)

    val documentAssembler = new MultiDocumentAssembler()
      .setInputCols("question", "context")
      .setOutputCols("document_question", "document_context")

    val questionAnswering = RoBertaForQuestionAnswering
      .pretrained("roberta_base_qa_squad2", "en")
      .setInputCols(Array("document_question", "document_context"))
      .setOutputCol("answer")
      .setCaseSensitive(true)

    val pipeline = new Pipeline().setStages(Array(documentAssembler, questionAnswering))
    val result = pipeline.fit(data).transform(data)

    val answers = result.selectExpr("explode(answer.result) as r").as[String].collect()
    answers.foreach(a => println(s"ANSWER=[$a]"))
    assert(answers.nonEmpty)
    answers.foreach { answer =>
      assert(!answer.contains(" ' "), s"stray space around apostrophe in '$answer'")
      assert(!answer.contains(" 's"), s"stray space before 's in '$answer'")
      assert(!answer.contains(" n't"), s"stray space before n't in '$answer'")
    }
  }
}
