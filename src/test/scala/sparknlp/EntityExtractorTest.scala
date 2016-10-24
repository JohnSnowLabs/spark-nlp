package sparknlp

import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class EntityExtractorTest extends SparkTest {
  test("simple entity extraction test") {
    val docs = Seq(
      TestRow(Document(
        "id",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et " +
          "dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut " +
          "aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum " +
          "dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui " +
          "officia deserunt mollit anim id est laborum."
      ))
    )
    val tokenPattern = "[a-zA-Z]+|[0-9]+|\\p{Punct}"
    val entities = EntityExtractor.loadEntities(getClass.getResourceAsStream("/test-phrases.txt"), tokenPattern)
    val dataset = spark.createDataFrame(docs)
    println(dataset.schema)
    val tokenizer = new RegexTokenizer()
      .setDocumentCol("document")
      .setPattern(tokenPattern)
    val stemmer = new Stemmer()
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("token"))
    val normalizer = new Normalizer()
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("stem"))
    val entityExtractor = new EntityExtractor()
      .setDocumentCol("document")
      .setInputAnnotationCols(Array("ntoken"))
      .setMaxLen(4)
      .setEntities(entities)
    val processed = entityExtractor.transform(normalizer.transform(stemmer.transform(tokenizer.transform(dataset))))
    println(processed.schema)
    processed.collect().foreach {
      row =>
        val document = Document(row.getAs[Row](0))
        println(document)
        row.getSeq[Row](4)
          .map(Annotation(_))
          .foreach {
            case entity: Annotation if entity.aType == "entity" =>
              println(entity, document.text.substring(entity.begin, entity.end))
            case _ => ()
          }
    }
  }
}
