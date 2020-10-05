package com.johnsnowlabs.nlp.annotators

import com.johnsnowlabs.nlp.{Annotation, AnnotatorApproach, AnnotatorModel, AnnotatorType, DocumentAssembler, HasRecursiveFit, HasRecursiveTransform, SparkAccessor, HasSimpleAnnotate}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.Dataset
import org.scalatest._

abstract class SomeApproach(override val uid: String) extends AnnotatorApproach[SomeModel] with HasRecursiveFit[SomeModel] {
  override val description: String = "Some Approach"

  def this() = this("foo_uid")

  /** Annotator reference id. Used to identify elements in metadata or to refer to this annotator type */
  override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)
  override val outputAnnotatorType: AnnotatorType = "BAR"
}

abstract class SomeModel(override val uid: String) extends AnnotatorModel[SomeModel] with HasRecursiveTransform[SomeModel] {

  def this() = this("bar_uid")

  override def annotate(annotations: Seq[Annotation], recursivePipeline: PipelineModel): Seq[Annotation]

  override val inputAnnotatorTypes: Array[String]
  override val outputAnnotatorType: AnnotatorType = "BAR"
}

class RecursiveTestSpec extends FlatSpec {

  val spark = SparkAccessor.spark
  import spark.implicits._

  val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")

  val token = new Tokenizer().setInputCols("document").setOutputCol("token")

  val data = Seq("Peter is a good person").toDF("text")

  "Recursive Approach" should "receive previous annotator models in the pipeline" in {
    import com.johnsnowlabs.nlp.recursive._

    val some = new SomeApproach() {
      override def train(dataset: Dataset[_], recursivePipeline: Option[PipelineModel]): SomeModel = {
        assert(recursivePipeline.isDefined, "because train has not received recursive pipeline")
        assert(recursivePipeline.get.stages.length == 2, "because train has not received exactly one stage in the recursive")
        assert(recursivePipeline.get.stages.head.isInstanceOf[DocumentAssembler], "because train has not received a document assembler")
        assert(recursivePipeline.get.stages(1).isInstanceOf[TokenizerModel], "because train has not received a document assembler")

        new SomeModel() with HasSimpleAnnotate[SomeModel] {
          override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
            annotations
          }

          override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.TOKEN)

          override def annotate(annotations: Seq[Annotation], recursivePipeline: PipelineModel): Seq[Annotation] = {
            annotate(annotations)
          }
        }
      }
    }.setInputCols("token").setOutputCol("baaar")

    val recursivePipeline = new Pipeline().setStages(Array(document, token, some)).recursive

    recursivePipeline.fit(data)

    succeed
  }

  "Recursive Model" should "receive annotator models in the pipeline" in {
    import com.johnsnowlabs.nlp.recursive._

    val someModel = new SomeModel() with HasSimpleAnnotate[SomeModel] {
      override def annotate(annotations: Seq[Annotation], recursivePipeline: PipelineModel): Seq[Annotation] = {
        assert(annotations.nonEmpty, "because received no annotations to annotate")
        assert(annotations.map(_.annotatorType).toSet.size == 1, "because did not get exactly DOCUMENT type annotations")
        assert(recursivePipeline.stages.length == 2, "because inner recursive pipeline did not have exactly 2 previous annotators")
        assert(recursivePipeline.stages(1).isInstanceOf[TokenizerModel], "because second recursive annotator was not a tokenizer model")

        val result = recursivePipeline.stages(1) match {
          case t: TokenizerModel => t.annotate(annotations)
          case _ => fail("Could not pattern match a TokenizerModel !")
        }

        // re-tokenize document annotations input
        assert(result.map(_.result).length == 5, "because did not tokenize correctly into 5 tokens")

        result
      }
      override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
        throw new IllegalStateException("SomeModel does not accept annotate without recursion")
      }

      override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)

    }.setInputCols("document").setOutputCol("baaar")
    val pipeline = new Pipeline().setStages(Array(document, token, someModel))
    pipeline.fit(data).recursive.transform(data).show()
    succeed
  }

  "Lazy Recursive Model" should "be ignored and used correctly once called" in {
    import com.johnsnowlabs.nlp.recursive._

    val lazyTokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").setLazyAnnotator(true)

    val someModel = new SomeModel() with HasSimpleAnnotate[SomeModel] {

      override def annotate(annotations: Seq[Annotation], recursivePipeline: PipelineModel): Seq[Annotation] = {

        val result = recursivePipeline.stages(1) match {
          case t: TokenizerModel => t.annotate(annotations)
          case _ => fail("Could not pattern match a TokenizerModel !")
        }

        // re-tokenize document annotations input
        result
      }

      override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)

      override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
        throw new IllegalStateException("SomeModel does not accept annotate without recursion")
      }
    }.setInputCols("document").setOutputCol("baaar")
    val pipeline = new Pipeline().setStages(Array(document, lazyTokenizer, someModel))
    val output = pipeline.fit(data).recursive.transform(data)

    assert(output.schema.fields.length == 3)
    assert(output.schema.fields.map(_.name).sameElements(Array("text", "document", "baaar")))

    output.select("text", "document", "baaar").show()

    succeed
  }

  "Lazy Recursive Model" should "work well in LightPipeline" in {
    import com.johnsnowlabs.nlp.recursive._

    val lazyTokenizer = new Tokenizer().setInputCols("document").setOutputCol("token").setLazyAnnotator(true)

    val someModel = new SomeModel() with HasSimpleAnnotate[SomeModel] {
      override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
        throw new IllegalStateException("SomeModel does not accept annotate without recursion")
      }

      override def annotate(annotations: Seq[Annotation], recursivePipeline: PipelineModel): Seq[Annotation] = {
        val result = recursivePipeline.stages(1) match {
          case t: TokenizerModel => t.annotate(annotations)
          case _ => fail("Could not pattern match a TokenizerModel !")
        }

        // re-tokenize document annotations input
        result
      }

      override val inputAnnotatorTypes: Array[String] = Array(AnnotatorType.DOCUMENT)

    }.setInputCols("document").setOutputCol("baaar")
    val pipeline = new Pipeline().setStages(Array(document, lazyTokenizer, someModel))
    val output = pipeline.fit(data)

    val result = output.annotate("Peter is a good person")

    assert(result.keys.size == 2)
    assert(result.contains("baaar"))
    assert(result.apply("baaar").length == 5)

  }

}
