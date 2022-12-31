import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline

object NerCrfTraining extends App {

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

  val ner = new NerCrfApproach().
    setInputCols("document", "token", "pos").
    setOutputCol("ner").
    setLabelColumn("label").
    setOutputCol("ner").
    setMinEpochs(1).
    setMaxEpochs(5).
    setEmbeddingsSource("data/embeddings/glove.6B.100d.txt", 100, WordEmbeddingsFormat.TEXT).
    setExternalFeatures("data/ner/dict.txt", ",").
    setExternalDataset("data/ner/eng.train", "SPARK_DATASET").
    setC0(1250000).
    setRandomSeed(0).
    setVerbose(2)

  val finisher = new Finisher().
    setInputCols("ner")

  val pipeline = new Pipeline().
    setStages(Array(
      documentAssembler,
      tokenizer,
      pos,
      ner,
      finisher
    ))

  val nermodel = pipeline.fit(Seq.empty[String].toDF("text"))
  val nerlpmodel = new LightPipeline(nermodel)

  val res = Benchmark.time("Light annotate NerCRF") {
    nerlpmodel.annotate("Peter is a very good person from Germany, he is working at IBM.")
  }

  println(res.mapValues(_.mkString(", ")).mkString(", "))

}

object NerDLTraining extends App {

  ResourceHelper.spark

  import ResourceHelper.spark.implicits._

  val documentAssembler = new DocumentAssembler().
    setInputCol("text").
    setOutputCol("document")

  val tokenizer = new Tokenizer().
    setInputCols(Array("document")).
    setOutputCol("token")

  val ner = new NerDLApproach().
    setInputCols("document", "token").
    setOutputCol("ner").
    setLabelColumn("label").
    setOutputCol("ner").
    setMinEpochs(1).
    setMaxEpochs(30).
    setEmbeddingsSource("data/embeddings/glove.6B.100d.txt", 100, WordEmbeddingsFormat.TEXT).
    setExternalDataset("data/ner/eng.train", "SPARK_DATASET").
    setRandomSeed(0).
    setVerbose(2).
    setDropout(0.8f).
    setBatchSize(18)

  val finisher = new Finisher().
    setInputCols("ner")

  val pipeline = new Pipeline().
    setStages(Array(
      documentAssembler,
      tokenizer,
      ner,
      finisher
    ))

  val nermodel = pipeline.fit(Seq.empty[String].toDF("text"))
  val nerlpmodel = new LightPipeline(nermodel)

  val res = Benchmark.time("Light annotate NerDL") {
    nerlpmodel.annotate("Peter is a very good person from Germany, he is working at IBM.")
  }

  println(res.mapValues(_.mkString(", ")).mkString(", "))

  nermodel.stages(2).asInstanceOf[NerDLModel].write.overwrite().save("./models/nerdl-deid-30")

}

object NerDLPretrained extends App {

  ResourceHelper.spark

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

  val ner = NerDLModel.pretrained().
    setInputCols("sentence", "token").
    setOutputCol("ner")

  val converter = new NerConverter()
    .setInputCols("sentence", "token", "ner")
    .setOutputCol("nerconverter")

  val finisher = new Finisher().
    setInputCols("token", "sentence", "nerconverter", "ner")

  val pipeline = new Pipeline().
    setStages(Array(
      documentAssembler,
      sentenceDetector,
      tokenizer,
      ner,
      converter,
      finisher
    ))

  val nermodel = pipeline.fit(Seq.empty[String].toDF("text"))
  val nerlpmodel = new LightPipeline(nermodel)

  val res1 = Benchmark.time("Light annotate NerDL") {
    nerlpmodel.fullAnnotate("Peter is a very good person from Germany, he is working at IBM.")
  }
  val res2 = Benchmark.time("Light annotate NerDL") {
    nerlpmodel.fullAnnotate("I saw the patient with Dr. Andrew Newhouse.")
  }
  val res3 = Benchmark.time("Light annotate NerDL") {
    nerlpmodel.fullAnnotate("Ms. Louise Iles is a 70 yearold")
  }
  val res4 = Benchmark.time("Light annotate NerDL") {
    nerlpmodel.fullAnnotate("Ms.")
  }

  println(res1.mapValues(_.mkString(", ")).mkString(", "))
  println(res2.mapValues(_.mkString(", ")).mkString(", "))
  println(res3.mapValues(_.mkString(", ")).mkString(", "))
  println(res4.mapValues(_.mkString(", ")).mkString(", "))

}

object NerCrfPretrained extends App {

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

  val ner = NerCrfModel.pretrained().
    setInputCols("sentence", "token", "pos").
    setOutputCol("ner")

  val converter = new NerConverter()
    .setInputCols("sentence", "token", "ner")
    .setOutputCol("nerconverter")

  val finisher = new Finisher().
    setInputCols("token", "sentence", "nerconverter", "ner")

  val pipeline = new Pipeline().
    setStages(Array(
      documentAssembler,
      sentenceDetector,
      tokenizer,
      pos,
      ner,
      converter,
      finisher
    ))

  val nermodel = pipeline.fit(Seq.empty[String].toDF("text"))
  val nerlpmodel = new LightPipeline(nermodel)

  val res1 = Benchmark.time("Light annotate NerCrf") {
    nerlpmodel.fullAnnotate("Peter is a very good person from Germany, he is working at IBM.")
  }
  val res2 = Benchmark.time("Light annotate NerCrf") {
    nerlpmodel.fullAnnotate("I saw the patient with Dr. Andrew Newhouse.")
  }
  val res3 = Benchmark.time("Light annotate NerCrf") {
    nerlpmodel.fullAnnotate("Ms. Louise Iles is a 70yearold")
  }
  val res4 = Benchmark.time("Light annotate NerCrf") {
    nerlpmodel.fullAnnotate("Ms.")
  }

  println(res1.mapValues(_.mkString(", ")).mkString(", "))
  println(res2.mapValues(_.mkString(", ")).mkString(", "))
  println(res3.mapValues(_.mkString(", ")).mkString(", "))
  println(res4.mapValues(_.mkString(", ")).mkString(", "))

}

