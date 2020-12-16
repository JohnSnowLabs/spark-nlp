package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.ml.tensorflow.TensorflowWrapper
import com.johnsnowlabs.ml.tensorflow.TensorflowT5
import com.johnsnowlabs.ml.tensorflow.sentencepiece.SentencePieceWrapper
import com.johnsnowlabs.nlp.annotator.SentenceDetectorDLModel
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.DocumentAssembler
import com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.scalatest._

import java.nio.file.{Files, Paths}
import scala.collection.mutable

class T5TestSpec extends FlatSpec {

  "T5 " should "run SparkNLP pipeline" in {
    val testData = ResourceHelper.spark.createDataFrame(Seq(

      (1, "Which is the capital of France? Who was the first president of USA?"),
      (1, "Which is the capital of Bulgaria?"),
      (2, "Who is Donald Trump?")

    )).toDF("id", "text")

    val documentAssembler = new DocumentAssembler()
      .setInputCol("text")
      .setOutputCol("questions")

    val sentenceDetector = SentenceDetectorDLModel
      .pretrained()
      .setInputCols(Array("questions"))
      .setOutputCol("question")

    val tokenizer = new Tokenizer()
      .setInputCols(Array("question"))
      .setOutputCol("tokens")

    val t5 = T5Transformer
      .load("/models/sparknlp/google_t5_small_ssm_nq")
      .setInputCols("tokens")
      .setOutputCol("answer")

    val pipeline = new Pipeline().setStages(Array(documentAssembler, sentenceDetector, tokenizer, t5))

    val model = pipeline.fit(testData)
    val results = model.transform(testData)

    results.select("question.result", "answer.result").show(truncate = false)
  }


  "T5Transformer" should "load and save model" ignore {
    val t5model = T5Transformer.loadSavedModel("/models/t5/google_t5_small_ssm_nq/tf/combined", ResourceHelper.spark)
    t5model.write.overwrite().save("/models/sparknlp/google_t5_small_ssm_nq")
  }

  "T5 TF" should "process text" ignore {
    val texts = Array("When was America discovered?", "Which was the first president of USA?")
    val tfw = TensorflowWrapper.read("/models/t5/google_t5_small_ssm_nq/tf/combined", zipped = false, useBundle = true, tags = Array("serve"))
    val spp = SentencePieceWrapper.read("/models/t5/google_t5_small_ssm_nq/tf/combined/assets/spiece.model")
    val t5tf = new TensorflowT5(tfw, spp)
    t5tf.process(texts).foreach(x => println(x))
  }
}
