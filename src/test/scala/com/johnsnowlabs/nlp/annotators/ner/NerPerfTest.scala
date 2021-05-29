package com.johnsnowlabs.nlp.annotators.ner

import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.ner.crf.NerCrfApproach
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import com.johnsnowlabs.util.Benchmark
import org.scalatest._

class NerPerfTest extends FlatSpec {

  "NerCRF Approach" should "be fast to train" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler().
      setInputCol("text").
      setOutputCol("document")

    val tokenizer = new Tokenizer().
      setInputCols(Array("document")).
      setOutputCol("token")

    val pos = PerceptronModel.pretrained().
      setInputCols("document", "token").
      setOutputCol("pos")

    val embeddings = new WordEmbeddings()
      .setInputCols("document", "token", "pos")
      .setOutputCol("embeddings")
      .setStoragePath("src/test/resources/ner-corpus/embeddings.100d.test.txt", "TEXT")
      .setDimension(100)

    val ner = new NerCrfApproach().
      setInputCols("document", "token", "pos", "embeddings").
      setOutputCol("ner").
      setLabelColumn("label").
      setOutputCol("ner").
      setMinEpochs(1).
      setMaxEpochs(5).
      setC0(1250000).
      setRandomSeed(0).
      setVerbose(2)

    val finisher = new Finisher().
      setInputCols("ner")

    val recursivePipeline = new RecursivePipeline().
      setStages(Array(
        documentAssembler,
        tokenizer,
        pos,
        embeddings,
        ner,
        finisher
      ))

    val nermodel = recursivePipeline.fit(Seq.empty[String].toDF("text"))
    val nerlpmodel = new LightPipeline(nermodel)

    val res = Benchmark.time("Light annotate NerCRF") {nerlpmodel.annotate("Peter is a very good person from Germany, he is working at IBM.")}

    println(res.mapValues(_.mkString(", ")).mkString(", "))

  }

  "NerDL Approach" should "be fast to train" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler().
      setInputCol("text").
      setOutputCol("document")

    val tokenizer = new Tokenizer().
      setInputCols(Array("document")).
      setOutputCol("token")

    val embeddings = new WordEmbeddings()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setStoragePath("./embeddings.bin", "BINARY")
      .setDimension(200)

    val ner = new NerDLApproach().
      setInputCols("document", "token", "embeddings")
      .setOutputCol("ner")
      .setLabelColumn("label")
      .setOutputCol("ner")
      .setMinEpochs(1)
      .setMaxEpochs(30)
      .setRandomSeed(0)
      .setVerbose(2)
      .setDropout(0.8f)
      .setBatchSize(18)
      .setGraphFolder("src/test/resources/graph/")


    val finisher = new Finisher().
      setInputCols("ner")

    val recursivePipeline = new RecursivePipeline().
      setStages(Array(
        documentAssembler,
        tokenizer,
        embeddings,
        ner,
        finisher
      ))

    val nermodel = recursivePipeline.fit(Seq.empty[String].toDF("text"))
    val nerlpmodel = new LightPipeline(nermodel)

    val res = Benchmark.time("Light annotate NerDL") {nerlpmodel.annotate("Peter is a very good person from Germany, he is working at IBM.")}

    println(res.mapValues(_.mkString(", ")).mkString(", "))

    nermodel.stages(2).asInstanceOf[NerDLModel].write.overwrite().save("./models/nerdl-deid-30")

  }

  "NerDL Model" should "label correctly" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler().
      setInputCol("text").
      setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setUseAbbreviations(false)

    val tokenizer = new Tokenizer().
      setInputCols(Array("sentence")).
      setOutputCol("token")

    val ner = NerDLModel.pretrained().//.load("./models/nerdl-deid-30").//.pretrained().
      setInputCols("sentence", "token").
      setOutputCol("ner")

    val converter = new NerConverter()
      .setInputCols("sentence", "token", "ner")
      .setOutputCol("nerconverter")

    val finisher = new Finisher().
      setInputCols("token", "sentence", "nerconverter", "ner")

    val recursivePipeline = new RecursivePipeline().
      setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        ner,
        converter,
        finisher
      ))

    val nermodel = recursivePipeline.fit(Seq.empty[String].toDF("text"))
    val nerlpmodel = new LightPipeline(nermodel)

    val res1 = Benchmark.time("Light annotate NerDL") {nerlpmodel.fullAnnotate("Peter is a very good person from Germany, he is working at IBM.")}
    val res2 = Benchmark.time("Light annotate NerDL") {nerlpmodel.fullAnnotate("I saw the patient with Dr. Andrew Newhouse.")}
    val res3 = Benchmark.time("Light annotate NerDL") {nerlpmodel.fullAnnotate("Ms. Louise Iles is a 70 yearold")}
    val res4 = Benchmark.time("Light annotate NerDL") {nerlpmodel.fullAnnotate("Ms.")}

    println(res1.mapValues(_.mkString(", ")).mkString(", "))
    println(res2.mapValues(_.mkString(", ")).mkString(", "))
    println(res3.mapValues(_.mkString(", ")).mkString(", "))
    println(res4.mapValues(_.mkString(", ")).mkString(", "))

  }

  "NerCRF Model" should "label correctly" taggedAs SlowTest in {

    import ResourceHelper.spark.implicits._

    val documentAssembler = new DocumentAssembler().
      setInputCol("text").
      setOutputCol("document")

    val sentenceDetector = new SentenceDetector()
      .setInputCols("document")
      .setOutputCol("sentence")
      .setUseAbbreviations(false)

    val tokenizer = new Tokenizer().
      setInputCols(Array("sentence")).
      setOutputCol("token")

    val pos = PerceptronModel.pretrained().
      setInputCols("document", "token").
      setOutputCol("pos")

    val word_embeddings = WordEmbeddingsModel.pretrained()
      .setInputCols("document", "token")
      .setOutputCol("embeddings")
      .setCaseSensitive(false)

    //document, token, pos, word_embeddings
    val ner = NerCrfModel.pretrained().
      setInputCols("sentence", "token", "pos", "word_embeddings").
      setOutputCol("ner")

    val converter = new NerConverter()
      .setInputCols("sentence", "token", "ner")
      .setOutputCol("nerconverter")

    val finisher = new Finisher().
      setInputCols("token", "sentence", "nerconverter", "ner")

    val recursivePipeline = new RecursivePipeline().
      setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        pos,
        word_embeddings,
        ner,
        converter,
        finisher
      ))

    val nermodel = recursivePipeline.fit(Seq.empty[String].toDF("text"))
    val nerlpmodel = new LightPipeline(nermodel)

    val res1 = Benchmark.time("Light annotate NerCrf") {nerlpmodel.fullAnnotate("Peter is a very good person from Germany, he is working at IBM.")}
    val res2 = Benchmark.time("Light annotate NerCrf") {nerlpmodel.fullAnnotate("I saw the patient with Dr. Andrew Newhouse.")}
    val res3 = Benchmark.time("Light annotate NerCrf") {nerlpmodel.fullAnnotate("Ms. Louise Iles is a 70yearold")}
    val res4 = Benchmark.time("Light annotate NerCrf") {nerlpmodel.fullAnnotate("Ms.")}

    println(res1.mapValues(_.mkString(", ")).mkString(", "))
    println(res2.mapValues(_.mkString(", ")).mkString(", "))
    println(res3.mapValues(_.mkString(", ")).mkString(", "))
    println(res4.mapValues(_.mkString(", ")).mkString(", "))

  }

}

