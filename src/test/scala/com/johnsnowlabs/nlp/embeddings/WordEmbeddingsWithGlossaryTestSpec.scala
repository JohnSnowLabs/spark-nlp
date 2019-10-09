package com.johnsnowlabs.nlp.embeddings

import com.johnsnowlabs.nlp.annotator.{NerConverter, NerDLModel, SentenceDetector}
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.base.{DocumentAssembler, RecursivePipeline}
import com.johnsnowlabs.nlp.embeddings.{WordEmbeddings, WordEmbeddingsFormat, WordEmbeddingsModel}
import org.scalatest.FlatSpec
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions._
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.ml.evaluation.ClusteringEvaluator
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable


class WordEmbeddingsWithGlossaryTestSpec extends FlatSpec {


  "ClinicalWordEmbeddings" should "deidentify better with a surname glossary" in {

    val spark: SparkSession = SparkSession
      .builder()
      .appName("test")
      .master("local")
      .config("spark.driver.memory", "4G")
      .getOrCreate()

    import spark.implicits._

    val srcDF = Seq(
      "John Smith",
      "Ionnidis Papaloudus")
      .toDF("text")

    val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
    val sentenceDetector: SentenceDetector = new SentenceDetector().setInputCols(Array("document")).setOutputCol("sentence")
    val tokenizer = new Tokenizer().setInputCols(Array("sentence")).setOutputCol("token")

    val clinical_sensitive_entities: NerDLModel = NerDLModel.load("/sandbox/jsl/cache_pretrained/deidentify_dl_en_2.0.2_2.4_1559669094458")
      .setInputCols(Array("sentence", "token", "embeddings"))
      .setOutputCol("ner")

    val ner_converter: NerConverter = new NerConverter()
      .setInputCols(Array("sentence", "token", "ner"))
      .setOutputCol("ner_chunk")

    // first pipeline without glossary so "Ionnidis Papaloudus will not be in the embeddings.
    val embeddings = WordEmbeddingsModel.load("/sandbox/jsl/cache_pretrained/embeddings_clinical_en_2.0.2_2.4_1558454742956")
    embeddings.setInputCols(Array("sentence", "token")).setOutputCol("embeddings")

    val pipeline = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings,
        clinical_sensitive_entities,
        ner_converter
      ))

    val myDF: DataFrame = pipeline.fit(srcDF).transform(srcDF)
    myDF.show(false)


    // Now embeddings with glossary. We need to create a completely new embeddings_with_glossary
    // otherwise will leverage in previously load from RocksDB and glossary will not be injected propertly.
    val embeddings_with_glossary = WordEmbeddingsModel.load("/sandbox/jsl/cache_pretrained/embeddings_clinical_en_2.0.2_2.4_1558454742956")
    embeddings_with_glossary.setInputCols(Array("sentence", "token")).setOutputCol("embeddings")

    // Now add glossary assigning the same vector for all new surnames
    val refVectorForSurnames: Array[Double] = Array(0.015133019734713971,0.00930821884176369,-0.013587907167804435,
      -0.024398496587990252,0.012198621770107158,-0.025502355007613282,0.030872762522395004,-0.021773086359066933,
      -0.007729996541967824,-0.06831770848841166,-0.06092272900653961,0.02274967998630956,-0.047445249804859334,
      -0.001038311912393655,0.05740769385263601,-0.04893180632317975,-0.00919586505840943,0.08630655002252892,
      -0.02255619929473948,0.044714379795058413,-0.16723122044427405,0.0022823189560180947,0.05257370206023169,
      0.037956436354633936,-0.07368793788821337,-0.09075132389784439,-0.12928401138329426,0.0615267652499502,
      0.0030267937950159804,0.02366752685754591,0.008575393336289126,-0.086661916552893,0.04618026115037377,
      0.0839460844693614,-0.007503291523836413,-0.07480816658145566,0.1979398851424898,0.07289842742447528,
      0.039615638274425484,0.10846895066141456,0.003695583162188626,-0.013323716313351443,0.017122667920529685,
      -0.046193967358086355,0.09358804999028437,-0.0023818994755032564,-0.015301876563618096,-0.0750863776290264,
      -0.020824827239560882,-0.006806438516245876,0.022078161408024984,-0.0022333383286114532,0.1769956792998201,
      0.07411225984191157,-0.01865887653843248,-0.07331574228328443,0.01892692100015937,0.033182838541240336,
      -0.020841341709417243,-0.13601866172021373,0.15821796749054395,0.009724575703152888,-0.10365459940707077,
      0.048950533780542684,-0.10564294429448805,0.0429188745635879,0.01538856405475298,0.10705476418546542,
      -0.024817229258484248,0.028594848730601998,-0.04881634208007741,0.061710321495304275,0.08560984372510874,
      -0.036142741266643984,-9.215704909225362E-4,-0.11823445437220538,-0.2644106291879444,0.04326757048730863,
      -0.15476596612295174,0.07012244388614322,0.06577789719988834,0.04839023930752359,-0.10718401660591946,
      -0.02227984971844981,-0.18717209863559162,0.10254983106313917,-0.06093182377082355,-0.06780633575298346,
      -0.031924928865956305,-0.04733231215065449,0.0021983229497459004,0.0865922214616719,0.03130657621027336,
      -0.10479788587656054,0.06397549965405372,0.10952333648796245,0.06343153839470367,0.0718984990342741,
      0.08220809778847632,0.015413820859189924,0.17583025757128978,0.08986342347468279,0.09528408277763253,
      -0.14564120898991134,0.15485197779394969,0.13142570707841797,0.04977477315403774,0.05652014631736564,
      0.11179204139039063,0.015070730080648638,0.04214471451804101,0.0658125345703803,-0.005577842702380731,
      0.009123770601624778,-0.16156703421188237,0.039115926356399586,0.18871662933827515,-0.22839815443351008,
      0.2398376590169233,0.03411408928363023,0.01009589974143576,-0.04717387956029548,0.0016196929739338373,
      -0.08765373131902189,0.1461727326493862,0.036564203912383735,0.08253700993985333,0.03520819031986533,
      0.05545495838913183,0.16631554037638294,0.03370078612161924,-0.06627087379226876,-0.06032130627302687,
      0.09360630461162874,-0.07712001007972873,-0.011039888644356061,0.03700276152696086,-0.17520127766604576,
      -0.10095682039619412,-0.11026998607639046,0.07170162238148682,-0.013378040776211469,-0.0728340247936319,
      0.135002270456061,-0.03877985774774203,0.09919918756402432,-0.09704196390991575,0.2173582484706666,
      -0.16551096690788875,-0.2628865205612597,-0.02383996815619274,-0.038208772570474246,-0.02311398788437007,
      0.057606951591560526,-0.031225621899022057,0.07088492876805447,-0.07424049508320807,-0.20441792679434673,
      -0.10090459989059582,0.05554887294834679,0.1455431035389441,-0.12616359466468272,-0.09710157391744932,
      -0.016504095964728956,0.02346040262583152,0.024559916274229557,0.0367627993854813,-0.1398137642967172,
      0.17377185595026645,-0.031288684394593966,-0.190744747357448,-0.09769177404011649,0.21399625883129536,
      -0.08141188519033937,0.012260450763286723,-0.025246786061588648,-0.05440402469127033,0.025199106336603043,
      -0.03976761843201431,-0.26971863603807145,-0.0011522514762467592,0.04174516238513965,0.14997429531641504,
      0.03471572885850109,-0.02909871590245672,-0.08953820817022762,0.1024993670912912,0.059773229783515076,
      0.14576204604792745,0.12304200332212635,0.03085837573131875,0.15162098552985667,-0.007397393027811027,
      0.09621070261235588,0.0687351546258203,0.1108294289014533,-0.1410105791897687,0.03382138453326967,
      0.009188345405022879,0.050132559881929765)

    val refVectorsForSurnamesFloat = refVectorForSurnames.map{myVal: Double => myVal.toFloat}

    //TODO: Load the list of surnames from a glossary that only contains the non-included surnames
    val myGlossary: Option[Map[String, Array[Float]]] = Some(Map(
      "Ionnidis" -> refVectorsForSurnamesFloat,
      "Papaloudus"-> refVectorsForSurnamesFloat))

    embeddings_with_glossary.setGlossary(myGlossary)

    val clinical_sensitive_entities_with_glossary: NerDLModel = NerDLModel.load("/sandbox/jsl/cache_pretrained/deidentify_dl_en_2.0.2_2.4_1559669094458")
      .setInputCols(Array("sentence", "token", "embeddings"))
      .setOutputCol("ner")

    val ner_converter_with_glossary: NerConverter = new NerConverter()
      .setInputCols(Array("sentence", "token", "ner"))
      .setOutputCol("ner_chunk")

    val pipeline_with_glossary = new RecursivePipeline()
      .setStages(Array(
        documentAssembler,
        sentenceDetector,
        tokenizer,
        embeddings_with_glossary,
        clinical_sensitive_entities_with_glossary,
        ner_converter_with_glossary
      ))

    val myDFWithGlossary = pipeline_with_glossary.fit(srcDF).transform(srcDF)
    myDFWithGlossary.show(false)

  }


  /*
 "WordEmbeddings" should "add a glossary" in {

   val spark: SparkSession = SparkSession
     .builder()
     .appName("test")
     .master("local")
     .config("spark.driver.memory", "4G")
     .getOrCreate()

   import spark.implicits._

   val srcDF = Seq(
     "John Smith",
     "Ionnidis Papaloudus")
     .toDF("text")

   val documentAssembler = new DocumentAssembler().setInputCol("text").setOutputCol("document")
   val tokenizer = new Tokenizer().setInputCols(Array("document")).setOutputCol("token")
   // first pipeline without glossary so "Ionnidis Papaloudus will not be in the embeddings.
   val glove_100d_embeddings = WordEmbeddingsModel.load("/sandbox/jsl/cache_pretrained/glove_100d_en_2.0.2_2.4_1556534397055")

   glove_100d_embeddings.setInputCols(Array("document", "token")).setOutputCol("embeddings")

   val pipeline = new RecursivePipeline()
     .setStages(Array(
       documentAssembler,
       tokenizer,
       glove_100d_embeddings
     ))

   val myDF: DataFrame = pipeline.fit(srcDF).transform(srcDF)
   myDF.show(false)

   val glove_100d_embeddings_with_glossary = WordEmbeddingsModel.load("/sandbox/jsl/cache_pretrained/glove_100d_en_2.0.2_2.4_1556534397055")
   glove_100d_embeddings_with_glossary.setInputCols(Array("document", "token")).setOutputCol("embeddings")
   // Now add glossary to glove_100d
   val myVector = Array.fill(100)("1".toFloat) // lets imagine an arbitrary Vector full of 1.0
   val myGlossary: Option[Map[String, Array[Float]]] = Some(Map("Ionnidis" -> myVector, "Papaloudus"-> myVector))
   glove_100d_embeddings_with_glossary.setGlossary(myGlossary)

   val pipeline_with_glossary = new RecursivePipeline()
     .setStages(Array(
       documentAssembler,
       tokenizer,
       glove_100d_embeddings_with_glossary
     ))

   val myDFWithGlossary = pipeline_with_glossary.fit(srcDF).transform(srcDF)
   myDFWithGlossary.show(false)

 }
*/




}
