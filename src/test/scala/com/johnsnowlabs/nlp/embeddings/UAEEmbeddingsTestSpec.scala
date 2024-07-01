package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.base.LightPipeline
import com.johnsnowlabs.nlp.{Annotation, DocumentAssembler, EmbeddingsFinisher}
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import com.johnsnowlabs.tags.SlowTest
import org.apache.spark.ml.Pipeline
import org.scalatest.flatspec.AnyFlatSpec
import com.johnsnowlabs.util.TestUtils.tolerantFloatEq

class UAEEmbeddingsTestSpec extends AnyFlatSpec {
  lazy val spark = ResourceHelper.spark
  import spark.implicits._
  behavior of "UAEEmbeddings"

  lazy val document = new DocumentAssembler().setInputCol("text").setOutputCol("document")
  lazy val model = UAEEmbeddings
    .pretrained()
    .setInputCols("document")
    .setOutputCol("embeddings")

  lazy val rawData: Seq[String] = Seq("hello world", "hello moon")
  lazy val data = rawData.toDF("text")
  lazy val embeddingsFinisher = new EmbeddingsFinisher()
    .setInputCols("embeddings")
    .setOutputCols("embeddings")

  lazy val pipeline = new Pipeline().setStages(Array(document, model, embeddingsFinisher))

  /** Asserts the first 16 values of the embeddings are within tolerance.
    *
    * @param expected
    *   The expected embeddings
    */
  private def assertEmbeddings(
      expected: Array[Array[Float]],
      pipeline: Pipeline = pipeline): Unit = {
    val result = pipeline.fit(data).transform(data)

    result.selectExpr("explode(embeddings)").show(5, 80)

    val extractedEmbeddings =
      result.selectExpr("explode(embeddings)").as[Array[Float]].collect()
    extractedEmbeddings
      .zip(expected)
      .foreach { case (embeddings, expected) =>
        embeddings.take(16).zip(expected).foreach { case (e, exp) =>
          assert(e === exp, "Embedding value not within tolerance")
        }
      }
  }

  it should "work with default (cls) pooling" taggedAs SlowTest in {
    val expected: Array[Array[Float]] = Array(
      Array(0.50387836f, 0.5861595f, 0.35129607f, -0.76046336f, -0.32446113f, -0.11767582f,
        0.49193293f, 0.58396333f, 0.8440052f, 0.3409165f, 0.02228897f, 0.3270517f, -0.3040624f,
        0.0651551f, -0.7069445f, 0.39551276f),
      Array(0.66606593f, 0.9617606f, 0.24854378f, -0.10180531f, -0.6569206f, 0.02763455f,
        0.19156311f, 0.7743124f, 1.0966388f, -0.03704539f, 0.43159822f, 0.48135376f, -0.47491387f,
        -0.22510622f, -0.7761906f, -0.29289678f))
    assertEmbeddings(expected)
  }

  val expected_cls_avg = Array(
    Array(0.42190665f, 0.48439154f, 0.37961221f, -0.88345671f, -0.39864743f, -0.10434269f,
      0.47246569f, 0.57266355f, 0.90948695f, 0.34240869f, -0.05249403f, 0.20690459f, -0.2502915f,
      -0.075280815f, -0.72355306f, 0.37840521f),
    Array(0.61534011f, 0.86877286f, 0.30440071f, -0.11193186f, -0.64877027f, 0.03778841f,
      0.19575913f, 0.77637982f, 1.0544734f, 0.02276843f, 0.40709749f, 0.48178568f, -0.45722729f,
      -0.25922f, -0.75728685f, -0.2886759f))

  it should "work with cls_avg pooling" taggedAs SlowTest in {
    model.setPoolingStrategy("cls_avg")
    assertEmbeddings(expected_cls_avg)
  }

  it should "work with last pooling" taggedAs SlowTest in {
    model.setPoolingStrategy("last")
    val expected = Array(
      Array(0.32610807f, 0.40477207f, 0.5753994f, -1.0180508f, -0.15669955f, -0.26589864f,
        0.57111073f, 0.59625691f, 0.98112649f, 0.31161842f, -0.088124298f, -0.23382883f,
        -0.10615025f, -0.4932569f, -0.92297047f, 0.64136416f),
      Array(0.42494205f, 0.91639936f, 0.47431907f, -0.11696267f, -0.78128248f, -0.044441216f,
        0.34416255f, 0.91160774f, 1.0371225f, 0.28027025f, 0.49664021f, 0.60586137f, -0.52690864f,
        -0.49278158f, -1.0315861f, -0.10492325f))
    assertEmbeddings(expected)
  }

  it should "work with avg pooling" taggedAs SlowTest in {
    model.setPoolingStrategy("avg")
    val expected = Array(
      Array(0.33993506f, 0.38262373f, 0.40792847f, -1.0064504f, -0.47283337f, -0.091009863f,
        0.45299777f, 0.5613634f, 0.97496814f, 0.34390116f, -0.12727717f, 0.086757362f,
        -0.19652022f, -0.21571696f, -0.740161f, 0.36129794f),
      Array(0.5646143f, 0.77578509f, 0.36025763f, -0.12205841f, -0.64061993f, 0.047942273f,
        0.19995515f, 0.77844721f, 1.0123079f, 0.08258225f, 0.38259676f, 0.48221761f, -0.43954074f,
        -0.2933338f, -0.73838311f, -0.28445506f))
    assertEmbeddings(expected)
  }
  it should "work with max pooling" taggedAs SlowTest in {
    model.setPoolingStrategy("max")
    val expected = Array(
      Array(0.50387824f, 0.58615935f, 0.5753994f, -0.76046306f, -0.15669955f, 0.070831679f,
        0.57988632f, 0.63754135f, 1.0989035f, 0.36707285f, 0.022289103f, 0.32705182f,
        -0.094303429f, 0.065155327f, -0.59403443f, 0.64136416f),
      Array(0.69840318f, 0.96176058f, 0.47431907f, -0.053866591f, -0.49888393f, 0.36105314f,
        0.34416255f, 0.91160774f, 1.192958f, 0.28027025f, 0.49664021f, 0.60586137f, -0.31200063f,
        -0.21072304f, -0.46940672f, -0.10492325f))
    assertEmbeddings(expected)
  }
  it should "work with integer pooling" taggedAs SlowTest in {
    model.setPoolingStrategy("2")
    val expected = Array(
      Array(0.13630758f, 0.26152137f, 0.13758762f, -1.2564588f, -0.8082003f, 0.070831679f,
        0.16906039f, 0.42769182f, 1.0989035f, 0.36707285f, -0.056193497f, 0.27165085f,
        -0.094303429f, -0.38840955f, -0.73669398f, 0.21443801f),
      Array(0.69840318f, 0.54228693f, 0.29342332f, -0.21559906f, -0.49888393f, 0.36105314f,
        0.14411977f, 0.52433759f, 0.72251248f, 0.039639104f, 0.37450147f, 0.59273022f,
        -0.31200063f, -0.21072304f, -0.46940672f, -0.45673683f))
    assertEmbeddings(expected)
  }

  it should "be compatible with LightPipeline" taggedAs SlowTest in {
    model.setPoolingStrategy("cls_avg")
    val pipelineModel = pipeline.fit(data)
    val lightPipeline = new LightPipeline(pipelineModel)
    val result = lightPipeline.fullAnnotate(rawData.toArray)

    val extractedEmbeddings: Array[Array[Float]] =
      result.map(_("embeddings").head.asInstanceOf[Annotation].embeddings)
    extractedEmbeddings
      .zip(expected_cls_avg)
      .foreach { case (embeddings, expected) =>
        embeddings.take(16).zip(expected).foreach { case (e, exp) =>
          assert(e === exp, "Embedding value not within tolerance")
        }
      }
  }

  it should "be serializable" taggedAs SlowTest in {
    model.setPoolingStrategy("cls_avg")
    val pipelineModel = pipeline.fit(data)
    pipelineModel
      .stages(1)
      .asInstanceOf[UAEEmbeddings]
      .write
      .overwrite()
      .save("./tmp_uae_model")

    val loadedModel = UAEEmbeddings.load("./tmp_uae_model")
    val newPipeline: Pipeline =
      new Pipeline().setStages(Array(document, loadedModel, embeddingsFinisher))

    assertEmbeddings(expected_cls_avg, newPipeline)
  }

}
