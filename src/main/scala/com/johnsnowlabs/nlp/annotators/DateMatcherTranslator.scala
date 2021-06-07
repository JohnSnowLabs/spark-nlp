package com.johnsnowlabs.nlp.annotators

object DateMatcherTranslator extends Serializable {

  val Language = "it"
  val DictionaryPath = s"src/main/resources/date-matcher/date_translation_data/$Language.json"
//  val DictionaryPath = s"/home/wolliqeonii/workspace/dev/jsl/spark-nlp/src/main/resources/date-matcher/date_translation_data/$Language.json"

  def loadTranslationDictionary() = {

    import org.json4s._
    import org.json4s.jackson.JsonMethods._

    import scala.io.Source

    // reading a file
    val jsonString = Source.fromFile(DictionaryPath).mkString
    println(jsonString)

    val json = parse(jsonString)

//    println(json)

    // Converting from JOjbect to plain object
    implicit val formats = DefaultFormats
    val myOldMap = json.extract[Map[String, Any]]

    println(myOldMap)
  }

  val dictionary = loadTranslationDictionary()

//  val dictionary = SparkSession.builder().getOrCreate()
//    .read
//    .option("multiline", true)
//    //      .option("mode", "PERMISSIVE")
//    .json(DictionaryPath)

  def searchLanguage(text: String) = {
    //    val dictionary = JSON.parseFull(DictionaryPath)

//    val url = ClassLoader.getSystemResource(DictionaryPath)
//    val schemaSource = Source.fromFile(url.getFile).getLines.mkString
//    val schemaFromJson = DataType.fromJson(schemaSource).asInstanceOf[StructType]
//    val dictionary = spark.read.schema(schemaFromJson)
//      .json(DictionaryPath)
//    dictionary.printSchema()
//    dictionary.show(false)
//
//    dictionary.show()
  }

  def detectLanguage(text: String, sourceLanguage: String) = {
    println(s"Ciao, detected language: $sourceLanguage")

    searchLanguage(text)
    text
  }

  def translateLanguage(text: String, sourceLanguage: String, destinationLanguage: String = "en") = {
    println("Detecting...")
    val detected = detectLanguage(text, sourceLanguage)
    println("Translating...")
    detected
  }
}
