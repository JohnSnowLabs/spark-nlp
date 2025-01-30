package com.johnsnowlabs.nlp.annotators.classifier.dl

import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.base.{LightPipeline, MultiDocumentAssembler}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalactic.TolerantNumerics
import org.scalatest.flatspec.AnyFlatSpec

class MPNetForQuestionAnsweringTestSpec extends AnyFlatSpec {
  val spark = ResourceHelper.spark
  import spark.implicits._

  lazy val document = new MultiDocumentAssembler()
    .setInputCols("question", "context")
    .setOutputCols("document_question", "document_context")

  lazy val questionAnswering = MPNetForQuestionAnswering
    .pretrained()
    .setInputCols(Array("document_question", "document_context"))
    .setOutputCol("answer")

  lazy val pipeline = new Pipeline().setStages(Array(document, questionAnswering))

  lazy val question = "Which name is also used to describe the Amazon rainforest in English?"
  lazy val context =
    "The Amazon rainforest (Portuguese: Floresta Amazônica or Amazônia; Spanish: Selva " +
      "Amazónica, Amazonía or usually Amazonia; French: Forêt amazonienne; Dutch: " +
      "Amazoneregenwoud), also known in English as Amazonia or the Amazon Jungle, is a moist " +
      "broadleaf forest that covers most of the Amazon basin of South America. This basin " +
      "encompasses 7,000,000 square kilometres (2,700,000 sq mi), of which 5,500,000 square " +
      "kilometres (2,100,000 sq mi) are covered by the rainforest. This region includes " +
      "territory belonging to nine nations. The majority of the forest is contained within " +
      "Brazil, with 60% of the rainforest, followed by Peru with 13%, Colombia with 10%, and " +
      "with minor amounts in Venezuela, Ecuador, Bolivia, Guyana, Suriname and French Guiana." +
      " States or departments in four nations contain \"Amazonas\" in their names. The Amazon" +
      " represents over half of the planet's remaining rainforests, and comprises the largest" +
      " and most biodiverse tract of tropical rainforest in the world, with an estimated 390" +
      " billion individual trees divided into 16,000 species."

  lazy val data = Seq((question, context)).toDF("question", "context")

  lazy val expectedStart = 201
  lazy val expectedEnd = 230
  lazy val expectedAnswer = "Amazonia or the Amazon Jungle"
  lazy val expectedScore: Float = 0.09354283660650253f

  behavior of "MPNetForQuestionAnsweringTestSpec"

  it should "tokenize correctly" taggedAs SlowTest in {
    val expectedTokens = Array(0, 2033, 2175, 2007, 2040, 2113, 2004, 6239, 2000, 9737, 18955,
      2003, 2398, 1033, 2, 2, 2000, 9737, 18955, 1010, 5081, 1028, 17347, 2700, 9737, 5559, 2034,
      9737, 2405, 1029, 3013, 1028, 7371, 22148, 9737, 5559, 1014, 9737, 2405, 2034, 2792, 9737,
      2405, 1029, 2417, 1028, 18925, 2106, 9737, 9017, 2642, 1029, 3807, 1028, 9737, 7873, 6918,
      12159, 6788, 1011, 1014, 2040, 2128, 2003, 2398, 2008, 9737, 2405, 2034, 2000, 9737, 8898,
      1014, 2007, 1041, 11056, 5045, 19217, 3228, 2012, 4476, 2091, 2001, 2000, 9737, 6407, 2001,
      2152, 2641, 1016, 2027, 6407, 13978, 1025, 1014, 2203, 1014, 2203, 2679, 3721, 1010, 1020,
      1014, 6356, 1014, 2203, 5494, 2775, 1011, 1014, 2001, 2033, 1023, 1014, 3160, 1014, 2203,
      2679, 3721, 1010, 1020, 1014, 2535, 1014, 2203, 5494, 2775, 1011, 2028, 3143, 2015, 2000,
      18955, 1016, 2027, 2559, 2954, 3704, 7499, 2004, 3161, 3745, 1016, 2000, 3488, 2001, 2000,
      3228, 2007, 4842, 2310, 4384, 1014, 2011, 3442, 1007, 2001, 2000, 18955, 1014, 2632, 2015,
      7308, 2011, 2414, 1007, 1014, 7383, 2011, 2188, 1007, 1014, 2002, 2011, 3580, 8314, 2003,
      8330, 1014, 10382, 1014, 11649, 1014, 18790, 1014, 25054, 2002, 2417, 23572, 1016, 2167,
      2034, 7644, 2003, 2180, 3745, 5387, 1004, 9737, 3026, 1004, 2003, 2041, 3419, 1016, 2000,
      9737, 5840, 2062, 2435, 2001, 2000, 4778, 1009, 1059, 3592, 18955, 2019, 1014, 2002, 8685,
      2000, 2926, 2002, 2091, 16016, 4309, 16074, 12863, 2001, 5137, 18955, 2003, 2000, 2092,
      1014, 2011, 2023, 4362, 20028, 4555, 3269, 3632, 4059, 2050, 2389, 1014, 2203, 2431, 1016,
      2)

    val model = questionAnswering.getModelIfNotSet
    implicit def strToAnno(s: String): Annotation =
      Annotation("DOCUMENT", 0, s.length, s, Map.empty)

    val maxLength = 384
    val caseSensitive = false
    val questionTokenized =
      model.tokenizeDocument(
        docs = Seq(question),
        maxSeqLength = maxLength,
        caseSensitive = caseSensitive)

    val contextTokenized =
      model.tokenizeDocument(
        docs = Seq(context),
        maxSeqLength = maxLength,
        caseSensitive = caseSensitive)

    val tokenized = model.encodeSequence(questionTokenized, contextTokenized, maxLength).head
    assert(tokenized sameElements expectedTokens)
  }

  it should "predict correctly" taggedAs SlowTest in {
    val resultAnno = Annotation.collect(pipeline.fit(data).transform(data), "answer").head.head
    val (result, score, start, end) = (
      resultAnno.result,
      resultAnno.metadata("score").toFloat,
      resultAnno.metadata("start").toInt,
      resultAnno.metadata("end").toInt + 1)

    println(result, score)

    implicit val tolerantEq = TolerantNumerics.tolerantFloatEquality(1e-2f)
    assert(result == expectedAnswer, "Wrong Answer")
    assert(start == expectedStart, "Wrong start index")
    assert(end == expectedEnd, "Wrong end index")
    assert(score === expectedScore, "Wrong Score")
  }

  it should "work with multiple batches" taggedAs SlowTest in {
    val questions = Seq("What's my name?", "Where do I live?")
    val contexts =
      Seq("My name is Clara and I live in Berkeley.", "My name is Wolfgang and I live in Berlin.")

    val data = questions.zip(contexts).toDF("question", "context")
    pipeline.fit(data).transform(data).select("answer").show(false)
  }

  it should "be serializable" taggedAs SlowTest in {
    val pipelineModel = pipeline.fit(data)
    pipelineModel.stages.last
      .asInstanceOf[MPNetForQuestionAnswering]
      .write
      .overwrite()
      .save("./tmp_mpnet_qa")

    val loadedModel = MPNetForQuestionAnswering.load("./tmp_mpnet_qa")
    val newPipeline: Pipeline =
      new Pipeline().setStages(Array(document, loadedModel))

    val pipelineDF = newPipeline.fit(data).transform(data)

    val resultAnno = Annotation.collect(pipelineDF, "answer").head.head
    val (result, score, start, end) = (
      resultAnno.result,
      resultAnno.metadata("score").toFloat,
      resultAnno.metadata("start").toInt,
      resultAnno.metadata("end").toInt + 1)

    println(result, score)

    import com.johnsnowlabs.util.TestUtils.tolerantFloatEq
    assert(result == expectedAnswer, "Wrong Answer")
    assert(start == expectedStart, "Wrong start index")
    assert(end == expectedEnd, "Wrong end index")
    assert(score === expectedScore, "Wrong Score")
  }

  it should "be compatible with LightPipeline" taggedAs SlowTest in {
    val pipeline: Pipeline =
      new Pipeline().setStages(Array(document, questionAnswering))

    val pipelineModel = pipeline.fit(data)
    val lightPipeline = new LightPipeline(pipelineModel)
    val results = lightPipeline.fullAnnotate(Array(question), Array(context))

    results.foreach { result =>
      assert(result("document_question").nonEmpty)
      assert(result("document_context").nonEmpty)
      assert(result("answer").nonEmpty)
    }
  }
}
