package com.johnsnowlabs.pretrained
import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.SparkSession
/**
  * Combine a downloaded tokenizer with a locally created document assembler
  *
  * Created by jose on 22/02/18.
  */
object ModelDownloadSpec extends App {

  implicit val spark = SparkSession.builder().appName("i2b2 logreg").master("local[2]").getOrCreate
  import spark.implicits._

  val dataset = Seq("Songs are to be sung", "Dances are to be danced").toDF("text")

  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  val pipeline = new Pipeline().setStages(
    Array(documentAssembler, tokenizer)
  )
  print(pipeline.fit(dataset).transform(dataset))
}
