package sparknlp

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.scalatest.{BeforeAndAfterAll, FunSuite}

trait SparkTest extends FunSuite with BeforeAndAfterAll {

  var sc: SparkContext = _
  var sqlc: SQLContext = _

  override def beforeAll: Unit = {
    sc = new SparkContext(new SparkConf()
      .setMaster("local[2]")
      .setAppName("test"))
    sqlc = new SQLContext(sc)
  }

  override def afterAll: Unit = {
    sc.stop()
    sc = null
    sqlc = null
  }
}

case class TestRow(document: Document)