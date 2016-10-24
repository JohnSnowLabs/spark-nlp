package sparknlp

import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FunSuite}

trait SparkTest extends FunSuite with BeforeAndAfterAll {

  var spark: SparkSession = _

  override def beforeAll: Unit = {
    spark = SparkSession.builder()
      .master("local[2]")
      .appName("test")
      .getOrCreate()
  }

  override def afterAll: Unit = {
    spark.stop()
    spark = null
  }
}

case class TestRow(document: Document)