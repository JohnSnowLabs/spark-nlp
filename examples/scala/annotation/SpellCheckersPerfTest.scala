import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.annotators.spell.norvig.NorvigSweetingModel
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.util.Benchmark
import org.apache.spark.sql.functions.rand
import org.apache.spark.ml.Pipeline

class NorvigSweetingTest extends App {

  ResourceHelper.spark

  import ResourceHelper.spark.implicits._

  val documentAssembler = new DocumentAssembler().
    setInputCol("text").
    setOutputCol("document")

  val tokenizer = new Tokenizer().
    setInputCols(Array("document")).
    setOutputCol("token")

  val spell = NorvigSweetingModel.pretrained().
    setInputCols("token").
    setOutputCol("spell").
    setDoubleVariants(true)

  val finisher = new Finisher().
    setInputCols("spell")

  val pipeline = new Pipeline().
    setStages(Array(
      documentAssembler,
      tokenizer,
      spell,
      finisher
    ))

  val spellmodel = pipeline.fit(Seq.empty[String].toDF("text"))
  val spellplight = new LightPipeline(spellmodel)

  val n = 50

  val parquet = ResourceHelper.spark.read
    .text("data/vivekn/training_negative")
    .toDF("text").sort(rand())
  val data = parquet.as[String].take(n)
  data.length

  Benchmark.time("Light annotate norvig spell") {
    spellplight.annotate(data)
  }
}

class SymmetricDeleteTest extends App {

  ResourceHelper.spark

  import ResourceHelper.spark.implicits._

  val documentAssembler = new DocumentAssembler().
    setInputCol("text").
    setOutputCol("document")

  val tokenizer = new Tokenizer().
    setInputCols(Array("document")).
    setOutputCol("token")

  val spell = SymmetricDeleteModel.pretrained().
    setInputCols("token").
    setOutputCol("spell")

  val finisher = new Finisher().
    setInputCols("spell")

  val pipeline = new Pipeline().
    setStages(Array(
      documentAssembler,
      tokenizer,
      spell,
      finisher
    ))

  val spellmodel = pipeline.fit(Seq.empty[String].toDF("text"))
  val spellplight = new LightPipeline(spellmodel)

  val n = 50000

  val parquet = ResourceHelper.spark.read
    .text("data/vivekn/training_negative")
    .toDF("text").sort(rand())
  val data = parquet.as[String].take(n)
  data.length

  Benchmark.time("Light annotate symmetric spell") {
    spellplight.annotate(data)
  }
}
