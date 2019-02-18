import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.ner.NerConverter
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.embeddings.WordEmbeddingsFormat
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

object NerCrfExcludedEmbeddings extends App {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("test")
    .master("local[*]")
    .config("spark.driver.memory", "12G")
    .config("spark.kryoserializer.buffer.max","200M")
    .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  import spark.implicits._
  spark.sparkContext.setLogLevel("WARN")

  val data = Seq("Peter Parker is from New Zealand, and he is a wonderful man born in Germany, right in Berlin").toDS.toDF("text")

  val doc = new DocumentAssembler().
    setInputCol("text").
    setOutputCol("document")

  val sent = new SentenceDetector().
    setCustomBounds(Array(System.lineSeparator()+System.lineSeparator())).
    setInputCols(Array("document")).
    setOutputCol("sentence")

  val tok = new Tokenizer().
    setInputCols(Array("document")).
    setOutputCol("token")

  val pos = PerceptronModel.
    load("/home/saif/cache_pretrained/pos_fast_en_1.6.1_2_1533853928168/").
    setInputCols("token", "document").
    setOutputCol("pos")

  val ner = new NerCrfApproach().
    setInputCols("sentence", "token", "pos").
    setLabelColumn("label").
    /** https://github.com/patverga/torch-ner-nlp-from-scratch/raw/master/data/conll2003/eng.train */
    setExternalDataset("./eng.train").
    setC0(2250000).
    setRandomSeed(100).
    setMaxEpochs(1).
    setMinW(0.01).
    setOutputCol("ner").
    /** http://nlp.stanford.edu/data/glove.6B.zip */
    setEmbeddingsSource("./glove.6B.100d.txt", 100, "TEXT").
    setIncludeEmbeddings(false).
    setEmbeddingsRef("glove6b")

  val pipeline = new RecursivePipeline().setStages(
    Array(doc,
      sent,
      tok,
      pos,
      ner)).fit(data)

  pipeline.write.overwrite.save("pip_wo_embedd")

  val pipread = PipelineModel.load("pip_wo_embedd")

  import com.johnsnowlabs.nlp.embeddings.EmbeddingsHelper

  EmbeddingsHelper.load(
    "./glove.6B.100d.txt",
    spark,
    WordEmbeddingsFormat.TEXT.toString,
    "glove6b",
    200,
    true
  )

  pipread.transform(data).select("ner").show(false)

  pipread.stages(4).asInstanceOf[NerCrfModel].setIncludeEmbeddings(true).setEmbeddingsRef("glove6b")

  pipread.write.overwrite.save("pip_w_embeddings")

  val pipreadw = PipelineModel.load("pip_w_embeddings")

  pipreadw.transform(data).show()

}
