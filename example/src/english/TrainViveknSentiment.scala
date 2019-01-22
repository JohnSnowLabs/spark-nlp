import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession

object TrainViveknSentiment extends App {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("test")
    .master("local[*]")
    .config("spark.driver.memory", "4G")
    .config("spark.kryoserializer.buffer.max","200M")
    .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  spark.sparkContext.setLogLevel("WARN")

  import spark.implicits._

  val training = Seq(
    ("I really liked this movie!", "positive"),
    ("The cast was horrible", "negative"),
    ("Never going to watch this again or recommend it to anyone", "negative"),
    ("It's a waste of time", "negative"),
    ("I loved the protagonist", "positive"),
    ("The music was really really good", "positive")
  ).toDS.toDF("train_text", "train_sentiment")

  val testing = Array(
    "I don't recommend this movie, it's horrible",
    "Dont waste your time!!!"
  )

  val document = new DocumentAssembler()
    .setInputCol("train_text")
    .setOutputCol("document")

  val token = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

  val normalizer = new Normalizer()
    .setInputCols("token")
    .setOutputCol("normal")

  val vivekn = new ViveknSentimentApproach()
    .setInputCols("document", "normal")
    .setOutputCol("result_sentiment")
    .setSentimentCol("train_sentiment")

  val finisher = new Finisher()
    .setInputCols("result_sentiment")
    .setOutputCols("final_sentiment")

  val pipeline = new Pipeline().setStages(Array(document, token, normalizer, vivekn, finisher))

  val sparkPipeline = pipeline.fit(training)

  val lightPipeline = new LightPipeline(sparkPipeline)

  Benchmark.time("Light pipeline quick annotation") { lightPipeline.annotate(testing) }

  Benchmark.time("Spark pipeline, this may be too much for just two rows!") {
    val testingDS = testing.toSeq.toDS.toDF("testing_text")
    println("Updating DocumentAssembler input column")
    document.setInputCol("testing_text")
    sparkPipeline.transform(testingDS).show()
  }


}