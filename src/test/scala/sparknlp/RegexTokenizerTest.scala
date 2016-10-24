package sparknlp

import org.apache.spark.sql.Row
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner

@RunWith(classOf[JUnitRunner])
class RegexTokenizerTest extends SparkTest {
  test("simple tokenize test") {
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
    val dataset = spark.createDataFrame(docs)
    println(dataset.schema)
    val tokenizer = new RegexTokenizer()
      .setDocumentCol("document")
      .setPattern("[a-zA-Z]+|[0-9]+|\\p{Punct}")
    tokenizer.transform(dataset).collect().foreach {
      row =>
        val document = Document(row.getAs[Row](0))
        println(document)
        row.getSeq[Row](1).map(Annotation(_)).foreach {
          token =>
            println(token, document.text.substring(token.begin, token.end))
        }
    }
  }
}
