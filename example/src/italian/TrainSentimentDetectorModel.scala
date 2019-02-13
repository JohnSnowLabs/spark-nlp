import com.johnsnowlabs.nlp.{DocumentAssembler, Finisher}
import com.johnsnowlabs.nlp.annotators.{Normalizer, Stemmer, Tokenizer}
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.util.Benchmark
import com.johnsnowlabs.nlp.util.io.ExternalResource
import org.apache.spark.sql.SparkSession

object TrainSentimentDetectorModel extends App {

  val spark: SparkSession = SparkSession
    .builder()
    .appName("TrainLemmatizerModel")
    .master("local[*]")
    .config("spark.driver.memory", "12G")
    .config("spark.kryoserializer.buffer.max","200M")
    .config("spark.serializer","org.apache.spark.serializer.KryoSerializer")
    .getOrCreate()

  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val sentenceDetector = new SentenceDetector()
    .setInputCols(Array("document"))
    .setOutputCol("sentence")

  val tokenizer = new Tokenizer()
    .setInputCols(Array("sentence"))
    .setOutputCol("token")

  val normalizer = new Normalizer()
    .setInputCols("token")
    .setOutputCol("normal")

  /*
  * Here how you can download the dataset used in this example:
  * https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/it/lemma/dxc.technology/lemma_italian.txt
  * */
  val lemmatizer = new Lemmatizer()
    .setInputCols("normal")
    .setOutputCol("lemma")
    .setDictionary(
      ExternalResource(
        path = "/tmp/dxc.technology/lemma_italian.txt",
        readAs = "LINE_BY_LINE",
        Map("valueDelimiter" -> "\\s+" , "keyDelimiter" -> "->")
      )
    )

  /*
  * Here how you can download the dataset used in this example:
  * https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/resources/it/sentiment/dxc.technology/sentiment_italian.txt
  * */
  val sentiment = new SentimentDetector()
    .setInputCols("lemma","sentence")
    .setOutputCol("sentiment_score")
    .setDecrementMultiplier(0.5)
    .setIncrementMultiplier(1.5)
    .setDictionary(
      ExternalResource(
        "/tmp/dxc.technology/sentiment_italian.txt",
        readAs = "LINE_BY_LINE",
        Map("delimiter" -> ",")
      )
    )

  val pipeline = new Pipeline()
    .setStages(Array(
      documentAssembler,
      sentenceDetector,
      tokenizer,
      normalizer,
      lemmatizer,
      sentiment
    ))

  val sparkNLPPipelineModel = Benchmark.time("Time to train Spark NLP Pipeline!") {
    pipeline.fit(Seq.empty[String].toDF("text"))
  }

  // Test the Pipeline prediction

  // Create a DataFrame from some Italian text to test our Pipeline prediction
  val dfTest = List(
    "Finchè non avevo la linea ADSL di fastweb potevo entrare nel router e configurare quelle pochissime cose configurabili (es. nome dei device), da ieri che ho avuto la linea niente è più configurabile...",
    " L'uomo è insoddisfatto del prodotto.",
    " La coppia contenta si abbraccia sulla spiaggia.")
    .toDF("text")
  val pipeLinePredictionDF = sparkNLPPipelineModel.transform(dfTest)

  /* Obviously you can select multiple columns at the same time, but let's do it this way
   to display each annotator without truncating the their results
  * */
  pipeLinePredictionDF.select($"token.result".as("tokens")).show(false)
  pipeLinePredictionDF.select($"normal.result".as("normalized")).show(false)
  pipeLinePredictionDF.select($"lemma.result".as("lemmatized")).show(false)
  pipeLinePredictionDF.select($"sentiment_score.result".as("sentiment_score")).show(false)

}
