import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.SparkSession

object TokenizerWithNGram extends App {

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

  val document = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val token = new Tokenizer()
    .setInputCols("document")
    .setOutputCol("token")

  val normalizer = new Normalizer()
    .setInputCols("token")
    .setOutputCol("normal")

  val finisher = new Finisher()
    .setInputCols("normal")

  val ngram = new NGram()
    .setN(3)
    .setInputCol("finished_normal")
    .setOutputCol("3-gram")

  val gramAssembler = new DocumentAssembler()
    .setInputCol("3-gram")
    .setOutputCol("3-grams")

  val pipeline = new Pipeline().setStages(Array(document, token, normalizer, finisher, ngram, gramAssembler))

  val testing = Seq(
    (1, "Google is a famous company"),
    (2, "Peter Parker is a super heroe")
  ).toDS.toDF( "_id", "text")

  val result = pipeline.fit(Seq.empty[String].toDS.toDF("text")).transform(testing)
  Benchmark.time("Time to convert and show") {result.show(truncate=false)}


}
