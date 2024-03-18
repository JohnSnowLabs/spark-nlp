/*
 * Copyright 2017-2024 John Snow Labs
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.johnsnowlabs.nlp.annotators.seq2seq

import com.johnsnowlabs.ml.ai.util.Generation.GenerationConfig
import com.johnsnowlabs.ml.ai.NLLB
import com.johnsnowlabs.ml.onnx.OnnxWrapper.EncoderDecoderWithoutPastWrappers
import com.johnsnowlabs.ml.onnx.{OnnxWrapper, ReadOnnxModel, WriteOnnxModel}
import com.johnsnowlabs.ml.util.LoadExternalModel.{
  loadJsonStringAsset,
  loadSentencePieceAsset,
  modelSanityCheck,
  notSupportedEngineError
}
import com.johnsnowlabs.ml.util.ONNX
import com.johnsnowlabs.nlp.AnnotatorType.DOCUMENT
import com.johnsnowlabs.nlp._
import com.johnsnowlabs.ml.tensorflow.sentencepiece.{
  ReadSentencePieceModel,
  SentencePieceWrapper,
  WriteSentencePieceModel
}
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.SparkSession
import com.johnsnowlabs.nlp.serialization.{MapFeature, StructFeature}
import org.json4s._
import org.json4s.jackson.JsonMethods._

/** NLLB : multilingual translation model
  *
  * NLLB is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many
  * multilingual translation.
  *
  * The model can directly translate between the 9,900 directions of 100 languages.
  *
  * Pretrained models can be loaded with `pretrained` of the companion object:
  * {{{
  * val nllb = NLLBTransformer.pretrained()
  *   .setInputCols("document")
  *   .setOutputCol("generation")
  * }}}
  * The default model is `"nllb_418M"`, if no name is provided. For available pretrained models
  * please see the [[https://sparknlp.org/models?q=nllb Models Hub]].
  *
  * For extended examples of usage, see
  * [[https://github.com/JohnSnowLabs/spark-nlp/blob/master/src/test/scala/com/johnsnowlabs/nlp/annotators/seq2seq/NLLBTestSpec.scala NLLBTestSpec]].
  *
  * '''References:'''
  *   - [[https://arxiv.org/pdf/2207.04672.pdf No Language Left Behind: Scaling Human-Centered Machine Translation]]
  *   - [[https://github.com/facebookresearch/fairseq/tree/nllb]]
  *
  * '''Paper Abstract:'''
  *
  * ''Driven by the goal of eradicating language barriers on a global scale, machine translation
  * has solidified itself as a key focus of artificial intelligence research today. However, such
  * efforts have coalesced around a small subset of languages, leaving behind the vast majority of
  * mostly low-resource languages. What does it take to break the 200 language barrier while
  * ensuring safe, high quality results, all while keeping ethical considerations in mind? In No
  * Language Left Behind, we took on this challenge by first contextualizing the need for
  * low-resource language translation support through exploratory interviews with native speakers.
  * Then, we created datasets and models aimed at narrowing the performance gap between low and
  * high-resource languages. More specifically, we developed a conditional compute model based on
  * Sparsely Gated Mixture of Experts that is trained on data obtained with novel and effective
  * data mining techniques tailored for low-resource languages. We propose multiple architectural
  * and training improvements to counteract overfitting while training on thousands of tasks.
  * Critically, we evaluated the performance of over 40,000 different translation directions using
  * a human-translated benchmark, Flores-200, and combined human evaluation with a novel toxicity
  * benchmark covering all languages in Flores-200 to assess translation safety. Our model
  * achieves an improvement of 44% BLEU relative to the previous state-of-the-art, laying
  * important groundwork towards realizing a universal translation system. Finally, we open source
  * all contributions described in this work, accessible at this https URL. ''
  *
  * '''Languages Covered:'''
  *
  * Acehnese (Arabic script) (ace_Arab), Acehnese (Latin script) (ace_Latn), Mesopotamian Arabic
  * (acm_Arab), Ta’izzi-Adeni Arabic (acq_Arab), Tunisian Arabic (aeb_Arab), Afrikaans (afr_Latn),
  * South Levantine Arabic (ajp_Arab), Akan (aka_Latn), Amharic (amh_Ethi), North Levantine Arabic
  * (apc_Arab), Modern Standard Arabic (arb_Arab), Modern Standard Arabic (Romanized) (arb_Latn),
  * Najdi Arabic (ars_Arab), Moroccan Arabic (ary_Arab), Egyptian Arabic (arz_Arab), Assamese
  * (asm_Beng), Asturian (ast_Latn), Awadhi (awa_Deva), Central Aymara (ayr_Latn), South
  * Azerbaijani (azb_Arab), North Azerbaijani (azj_Latn), Bashkir (bak_Cyrl), Bambara (bam_Latn),
  * Balinese (ban_Latn), Belarusian (bel_Cyrl), Bemba (bem_Latn), Bengali (ben_Beng), Bhojpuri
  * (bho_Deva), Banjar (Arabic script) (bjn_Arab), Banjar (Latin script) (bjn_Latn), Standard
  * Tibetan (bod_Tibt), Bosnian (bos_Latn), Buginese (bug_Latn), Bulgarian (bul_Cyrl), Catalan
  * (cat_Latn), Cebuano (ceb_Latn), Czech (ces_Latn), Chokwe (cjk_Latn), Central Kurdish
  * (ckb_Arab), Crimean Tatar (crh_Latn), Welsh (cym_Latn), Danish (dan_Latn), German (deu_Latn),
  * Southwestern Dinka (dik_Latn), Dyula (dyu_Latn), Dzongkha (dzo_Tibt), Greek (ell_Grek),
  * English (eng_Latn), Esperanto (epo_Latn), Estonian (est_Latn), Basque (eus_Latn), Ewe
  * (ewe_Latn), Faroese (fao_Latn), Fijian (fij_Latn), Finnish (fin_Latn), Fon (fon_Latn), French
  * (fra_Latn), Friulian (fur_Latn), Nigerian Fulfulde (fuv_Latn), Scottish Gaelic (gla_Latn),
  * Irish (gle_Latn), Galician (glg_Latn), Guarani (grn_Latn), Gujarati (guj_Gujr), Haitian Creole
  * (hat_Latn), Hausa (hau_Latn), Hebrew (heb_Hebr), Hindi (hin_Deva), Chhattisgarhi (hne_Deva),
  * Croatian (hrv_Latn), Hungarian (hun_Latn), Armenian (hye_Armn), Igbo (ibo_Latn), Ilocano
  * (ilo_Latn), Indonesian (ind_Latn), Icelandic (isl_Latn), Italian (ita_Latn), Javanese
  * (jav_Latn), Japanese (jpn_Jpan), Kabyle (kab_Latn), Jingpho (kac_Latn), Kamba (kam_Latn),
  * Kannada (kan_Knda), Kashmiri (Arabic script) (kas_Arab), Kashmiri (Devanagari script)
  * (kas_Deva), Georgian (kat_Geor), Central Kanuri (Arabic script) (knc_Arab), Central Kanuri
  * (Latin script) (knc_Latn), Kazakh (kaz_Cyrl), Kabiyè (kbp_Latn), Kabuverdianu (kea_Latn),
  * Khmer (khm_Khmr), Kikuyu (kik_Latn), Kinyarwanda (kin_Latn), Kyrgyz (kir_Cyrl), Kimbundu
  * (kmb_Latn), Northern Kurdish (kmr_Latn), Kikongo (kon_Latn), Korean (kor_Hang), Lao
  * (lao_Laoo), Ligurian (lij_Latn), Limburgish (lim_Latn), Lingala (lin_Latn), Lithuanian
  * (lit_Latn), Lombard (lmo_Latn), Latgalian (ltg_Latn), Luxembourgish (ltz_Latn), Luba-Kasai
  * (lua_Latn), Ganda (lug_Latn), Luo (luo_Latn), Mizo (lus_Latn), Standard Latvian (lvs_Latn),
  * Magahi (mag_Deva), Maithili (mai_Deva), Malayalam (mal_Mlym), Marathi (mar_Deva), Minangkabau
  * (Arabic script) (min_Arab), Minangkabau (Latin script) (min_Latn), Macedonian (mkd_Cyrl),
  * Plateau Malagasy (plt_Latn), Maltese (mlt_Latn), Meitei (Bengali script) (mni_Beng), Halh
  * Mongolian (khk_Cyrl), Mossi (mos_Latn), Maori (mri_Latn), Burmese (mya_Mymr), Dutch
  * (nld_Latn), Norwegian Nynorsk (nno_Latn), Norwegian Bokmål (nob_Latn), Nepali (npi_Deva),
  * Northern Sotho (nso_Latn), Nuer (nus_Latn), Nyanja (nya_Latn), Occitan (oci_Latn), West
  * Central Oromo (gaz_Latn), Odia (ory_Orya), Pangasinan (pag_Latn), Eastern Panjabi (pan_Guru),
  * Papiamento (pap_Latn), Western Persian (pes_Arab), Polish (pol_Latn), Portuguese (por_Latn),
  * Dari (prs_Arab), Southern Pashto (pbt_Arab), Ayacucho Quechua (quy_Latn), Romanian (ron_Latn),
  * Rundi (run_Latn), Russian (rus_Cyrl), Sango (sag_Latn), Sanskrit (san_Deva), Santali
  * (sat_Olck), Sicilian (scn_Latn), Shan (shn_Mymr), Sinhala (sin_Sinh), Slovak (slk_Latn),
  * Slovenian (slv_Latn), Samoan (smo_Latn), Shona (sna_Latn), Sindhi (snd_Arab), Somali
  * (som_Latn), Southern Sotho (sot_Latn), Spanish (spa_Latn), Tosk Albanian (als_Latn), Sardinian
  * (srd_Latn), Serbian (srp_Cyrl), Swati (ssw_Latn), Sundanese (sun_Latn), Swedish (swe_Latn),
  * Swahili (swh_Latn), Silesian (szl_Latn), Tamil (tam_Taml), Tatar (tat_Cyrl), Telugu
  * (tel_Telu), Tajik (tgk_Cyrl), Tagalog (tgl_Latn), Thai (tha_Thai), Tigrinya (tir_Ethi),
  * Tamasheq (Latin script) (taq_Latn), Tamasheq (Tifinagh script) (taq_Tfng), Tok Pisin
  * (tpi_Latn), Tswana (tsn_Latn), Tsonga (tso_Latn), Turkmen (tuk_Latn), Tumbuka (tum_Latn),
  * Turkish (tur_Latn), Twi (twi_Latn), Central Atlas Tamazight (tzm_Tfng), Uyghur (uig_Arab),
  * Ukrainian (ukr_Cyrl), Umbundu (umb_Latn), Urdu (urd_Arab), Northern Uzbek (uzn_Latn), Venetian
  * (vec_Latn), Vietnamese (vie_Latn), Waray (war_Latn), Wolof (wol_Latn), Xhosa (xho_Latn),
  * Eastern Yiddish (ydd_Hebr), Yoruba (yor_Latn), Yue Chinese (yue_Hant), Chinese (Simplified)
  * (zho_Hans), Chinese (Traditional) (zho_Hant), Standard Malay (zsm_Latn), Zulu (zul_Latn).
  *
  * ==Example==
  * {{{
  * import spark.implicits._
  * import com.johnsnowlabs.nlp.base.DocumentAssembler
  * import com.johnsnowlabs.nlp.annotators.seq2seq.NLLBTransformer
  * import org.apache.spark.ml.Pipeline
  *
  * val documentAssembler = new DocumentAssembler()
  *   .setInputCol("text")
  *   .setOutputCol("documents")
  *
  * val nllb = NLLBTransformer.pretrained("nllb_418M")
  *   .setInputCols(Array("documents"))
  *   .setSrcLang("zh")
  *   .serTgtLang("en")
  *   .setMaxOutputLength(100)
  *   .setDoSample(false)
  *   .setOutputCol("generation")
  *
  * val pipeline = new Pipeline().setStages(Array(documentAssembler, nllb))
  *
  * val data = Seq(
  *   "生活就像一盒巧克力。"
  * ).toDF("text")
  * val result = pipeline.fit(data).transform(data)
  *
  * results.select("generation.result").show(truncate = false)
  * +-------------------------------------------------------------------------------------------+
  * |result                                                                                     |
  * +-------------------------------------------------------------------------------------------+
  * |[ Life is like a box of chocolate.]                                                        |
  * +-------------------------------------------------------------------------------------------+
  * }}}
  *
  * @param uid
  *   required uid for storing annotator to disk
  * @groupname anno Annotator types
  * @groupdesc anno
  *   Required input and expected output annotator types
  * @groupname Ungrouped Members
  * @groupname param Parameters
  * @groupname setParam Parameter setters
  * @groupname getParam Parameter getters
  * @groupname Ungrouped Members
  * @groupprio param  1
  * @groupprio anno  2
  * @groupprio Ungrouped 3
  * @groupprio setParam  4
  * @groupprio getParam  5
  * @groupdesc param
  *   A list of (hyper-)parameter keys this annotator can take. Users can set and get the
  *   parameter values through setters and getters, respectively.
  */
class NLLBTransformer(override val uid: String)
    extends AnnotatorModel[NLLBTransformer]
    with HasBatchedAnnotate[NLLBTransformer]
    with ParamsAndFeaturesWritable
    with WriteOnnxModel
    with HasGeneratorProperties
    with WriteSentencePieceModel
    with HasEngine {

  def this() = this(Identifiable.randomUID("NLLBTRANSFORMER"))

  /** Input annotator type : DOCUMENT
    *
    * @group param
    */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(DOCUMENT)

  /** Output annotator type : DOCUMENT
    *
    * @group param
    */
  override val outputAnnotatorType: String = DOCUMENT

  /** @group setParam */
  def setRandomSeed(value: Int): NLLBTransformer.this.type = {
    if (randomSeed.isEmpty) {
      this.randomSeed = Some(value)
    }
    this
  }

  /** A list of token ids which are ignored in the decoder's output (Default: `Array()`)
    *
    * @group param
    */
  var ignoreTokenIds = new IntArrayParam(
    this,
    "ignoreTokenIds",
    "A list of token ids which are ignored in the decoder's output")

  /** Source Language (Default: `en`)
    * @group param
    */
  var srcLang = new Param[String](this, "srcLang", "Source language")

  /** Target Language (Default: `fr`)
    * @group param
    */
  var tgtLang = new Param[String](this, "tgtLang", "Target language")

  def setSrcLang(value: String): NLLBTransformer.this.type = {
    val valueLower = value
    // check if language is supported
    if (!languageIds.contains(valueLower)) {
      throw new IllegalArgumentException(
        s"Language $value is not supported. Supported languages are: ${languageIds.mkString(", ")}")
    }
    srcLangToken = Some(languageIds.indexOf(valueLower))
    set(srcLang, valueLower)
  }

  def setTgtLang(value: String): NLLBTransformer.this.type = {
    val valueLower = value
    // check if language is supported
    if (!languageIds.contains(valueLower)) {
      throw new IllegalArgumentException(
        s"Language $value is not supported. Supported languages are: ${languageIds.mkString(", ")}")
    }
    tgtLangToken = Some(languageIds.indexOf(valueLower))
    set(tgtLang, value)
  }

  /** @group setParam */
  def setIgnoreTokenIds(tokenIds: Array[Int]): NLLBTransformer.this.type = {
    set(ignoreTokenIds, tokenIds)
  }

  /** @group getParam */
  def getIgnoreTokenIds: Array[Int] = $(ignoreTokenIds)

  def getSrcLangToken: Int = srcLangToken.getOrElse(languageIds.indexOf($(srcLang)))

  def getTgtLangToken: Int = tgtLangToken.getOrElse(languageIds.indexOf($(tgtLang)))

  private var _model: Option[Broadcast[NLLB]] = None
  private var srcLangToken: Option[Int] = None
  private var tgtLangToken: Option[Int] = None

  /** Vocabulary used to encode the words to ids with bpeTokenizer.encode
    *
    * @group param
    */
  val vocabulary: MapFeature[String, Int] = new MapFeature(this, "vocabulary").setProtected()

  /** @group setParam */
  def setVocabulary(value: Map[String, Int]): this.type = set(vocabulary, value)

  val generationConfig: StructFeature[GenerationConfig] =
    new StructFeature(this, "generationConfig").setProtected()

  def setGenerationConfig(value: GenerationConfig): this.type =
    set(generationConfig, value)

  def getGenerationConfig: GenerationConfig = $$(generationConfig)

  private val languageIds: Array[String] = Array(
    "ace_Arab",
    "ace_Latn",
    "acm_Arab",
    "acq_Arab",
    "aeb_Arab",
    "afr_Latn",
    "ajp_Arab",
    "aka_Latn",
    "amh_Ethi",
    "apc_Arab",
    "arb_Arab",
    "ars_Arab",
    "ary_Arab",
    "arz_Arab",
    "asm_Beng",
    "ast_Latn",
    "awa_Deva",
    "ayr_Latn",
    "azb_Arab",
    "azj_Latn",
    "bak_Cyrl",
    "bam_Latn",
    "ban_Latn",
    "bel_Cyrl",
    "bem_Latn",
    "ben_Beng",
    "bho_Deva",
    "bjn_Arab",
    "bjn_Latn",
    "bod_Tibt",
    "bos_Latn",
    "bug_Latn",
    "bul_Cyrl",
    "cat_Latn",
    "ceb_Latn",
    "ces_Latn",
    "cjk_Latn",
    "ckb_Arab",
    "crh_Latn",
    "cym_Latn",
    "dan_Latn",
    "deu_Latn",
    "dik_Latn",
    "dyu_Latn",
    "dzo_Tibt",
    "ell_Grek",
    "eng_Latn",
    "epo_Latn",
    "est_Latn",
    "eus_Latn",
    "ewe_Latn",
    "fao_Latn",
    "pes_Arab",
    "fij_Latn",
    "fin_Latn",
    "fon_Latn",
    "fra_Latn",
    "fur_Latn",
    "fuv_Latn",
    "gla_Latn",
    "gle_Latn",
    "glg_Latn",
    "grn_Latn",
    "guj_Gujr",
    "hat_Latn",
    "hau_Latn",
    "heb_Hebr",
    "hin_Deva",
    "hne_Deva",
    "hrv_Latn",
    "hun_Latn",
    "hye_Armn",
    "ibo_Latn",
    "ilo_Latn",
    "ind_Latn",
    "isl_Latn",
    "ita_Latn",
    "jav_Latn",
    "jpn_Jpan",
    "kab_Latn",
    "kac_Latn",
    "kam_Latn",
    "kan_Knda",
    "kas_Arab",
    "kas_Deva",
    "kat_Geor",
    "knc_Arab",
    "knc_Latn",
    "kaz_Cyrl",
    "kbp_Latn",
    "kea_Latn",
    "khm_Khmr",
    "kik_Latn",
    "kin_Latn",
    "kir_Cyrl",
    "kmb_Latn",
    "kon_Latn",
    "kor_Hang",
    "kmr_Latn",
    "lao_Laoo",
    "lvs_Latn",
    "lij_Latn",
    "lim_Latn",
    "lin_Latn",
    "lit_Latn",
    "lmo_Latn",
    "ltg_Latn",
    "ltz_Latn",
    "lua_Latn",
    "lug_Latn",
    "luo_Latn",
    "lus_Latn",
    "mag_Deva",
    "mai_Deva",
    "mal_Mlym",
    "mar_Deva",
    "min_Latn",
    "mkd_Cyrl",
    "plt_Latn",
    "mlt_Latn",
    "mni_Beng",
    "khk_Cyrl",
    "mos_Latn",
    "mri_Latn",
    "zsm_Latn",
    "mya_Mymr",
    "nld_Latn",
    "nno_Latn",
    "nob_Latn",
    "npi_Deva",
    "nso_Latn",
    "nus_Latn",
    "nya_Latn",
    "oci_Latn",
    "gaz_Latn",
    "ory_Orya",
    "pag_Latn",
    "pan_Guru",
    "pap_Latn",
    "pol_Latn",
    "por_Latn",
    "prs_Arab",
    "pbt_Arab",
    "quy_Latn",
    "ron_Latn",
    "run_Latn",
    "rus_Cyrl",
    "sag_Latn",
    "san_Deva",
    "sat_Beng",
    "scn_Latn",
    "shn_Mymr",
    "sin_Sinh",
    "slk_Latn",
    "slv_Latn",
    "smo_Latn",
    "sna_Latn",
    "snd_Arab",
    "som_Latn",
    "sot_Latn",
    "spa_Latn",
    "als_Latn",
    "srd_Latn",
    "srp_Cyrl",
    "ssw_Latn",
    "sun_Latn",
    "swe_Latn",
    "swh_Latn",
    "szl_Latn",
    "tam_Taml",
    "tat_Cyrl",
    "tel_Telu",
    "tgk_Cyrl",
    "tgl_Latn",
    "tha_Thai",
    "tir_Ethi",
    "taq_Latn",
    "taq_Tfng",
    "tpi_Latn",
    "tsn_Latn",
    "tso_Latn",
    "tuk_Latn",
    "tum_Latn",
    "tur_Latn",
    "twi_Latn",
    "tzm_Tfng",
    "uig_Arab",
    "ukr_Cyrl",
    "umb_Latn",
    "urd_Arab",
    "uzn_Latn",
    "vec_Latn",
    "vie_Latn",
    "war_Latn",
    "wol_Latn",
    "xho_Latn",
    "ydd_Hebr",
    "yor_Latn",
    "yue_Hant",
    "zho_Hans",
    "zho_Hant",
    "zul_Latn")

  /** @group setParam */
  def setModelIfNotSet(
      spark: SparkSession,
      onnxWrappers: EncoderDecoderWithoutPastWrappers,
      spp: SentencePieceWrapper): this.type = {
    if (_model.isEmpty) {
      _model = Some(
        spark.sparkContext.broadcast(
          new NLLB(
            onnxWrappers,
            spp = spp,
            generationConfig = getGenerationConfig,
            vocab = $$(vocabulary))))
    }
    this
  }

  /** @group getParam */
  def getModelIfNotSet: NLLB = _model.get.value

  setDefault(
    minOutputLength -> 10,
    maxOutputLength -> 200,
    doSample -> false,
    temperature -> 1.0,
    topK -> 50,
    topP -> 1.0,
    repetitionPenalty -> 1.0,
    noRepeatNgramSize -> 3,
    ignoreTokenIds -> Array(),
    batchSize -> 1,
    beamSize -> 1,
    maxInputLength -> 1024,
    srcLang -> "eng_Latn",
    tgtLang -> "fra_Latn")

  /** takes a document and annotations and produces new annotations of this annotator's annotation
    * type
    *
    * @param batchedAnnotations
    *   Annotations that correspond to inputAnnotationCols generated by previous annotators if any
    * @return
    *   any number of annotations processed for every input annotation. Not necessary one to one
    *   relationship
    */
  override def batchAnnotate(batchedAnnotations: Seq[Array[Annotation]]): Seq[Seq[Annotation]] = {

    val allAnnotations = batchedAnnotations
      .filter(_.nonEmpty)
      .zipWithIndex
      .flatMap { case (annotations, i) =>
        annotations.filter(_.result.nonEmpty).map(x => (x, i))
      }
    val processedAnnotations = if (allAnnotations.nonEmpty) {
      this.getModelIfNotSet.predict(
        sentences = allAnnotations.map(_._1),
        batchSize = $(batchSize),
        minOutputLength = $(minOutputLength),
        maxOutputLength = $(maxOutputLength),
        doSample = $(doSample),
        temperature = $(temperature),
        topK = $(topK),
        topP = $(topP),
        repetitionPenalty = $(repetitionPenalty),
        noRepeatNgramSize = $(noRepeatNgramSize),
        randomSeed = this.randomSeed,
        ignoreTokenIds = $(ignoreTokenIds),
        beamSize = $(beamSize),
        maxInputLength = $(maxInputLength),
        srcLangToken = getSrcLangToken,
        tgtLangToken = getTgtLangToken)
    } else {
      Seq()
    }
    Seq(processedAnnotations)
  }

  override def onWrite(path: String, spark: SparkSession): Unit = {
    super.onWrite(path, spark)
    getEngine match {
      case ONNX.name =>
        val wrappers = getModelIfNotSet.onnxWrappers
        val obj = getModelIfNotSet
        writeOnnxModels(
          path,
          spark,
          Seq((wrappers.encoder, "encoder_model.onnx")),
          NLLBTransformer.suffix)
        writeOnnxModels(
          path,
          spark,
          Seq((wrappers.decoder, "decoder_model.onnx")),
          NLLBTransformer.suffix)
        writeSentencePieceModel(
          path,
          spark,
          obj.spp,
          NLLBTransformer.suffix,
          NLLBTransformer.sppFile)
    }
  }
}

trait ReadablePretrainedNLLBTransformerModel
    extends ParamsAndFeaturesReadable[NLLBTransformer]
    with HasPretrained[NLLBTransformer] {
  override val defaultModelName: Some[String] = Some("nllb_418M")
  override val defaultLang: String = "xx"

  /** Java compliant-overrides */
  override def pretrained(): NLLBTransformer = super.pretrained()

  override def pretrained(name: String): NLLBTransformer = super.pretrained(name)

  override def pretrained(name: String, lang: String): NLLBTransformer =
    super.pretrained(name, lang)

  override def pretrained(name: String, lang: String, remoteLoc: String): NLLBTransformer =
    super.pretrained(name, lang, remoteLoc)
}

trait ReadNLLBTransformerDLModel extends ReadOnnxModel with ReadSentencePieceModel {
  this: ParamsAndFeaturesReadable[NLLBTransformer] =>

  override val onnxFile: String = "nllb_onnx"
  val suffix: String = "_nllb"
  override val sppFile: String = "nllb_spp"

  def readModel(instance: NLLBTransformer, path: String, spark: SparkSession): Unit = {
    instance.getEngine match {
      case ONNX.name =>
        val decoderWrappers =
          readOnnxModels(path, spark, Seq("decoder_model.onnx"), suffix)
        val encoderWrappers =
          readOnnxModels(path, spark, Seq("encoder_model.onnx"), suffix)
        val onnxWrappers =
          EncoderDecoderWithoutPastWrappers(
            decoder = decoderWrappers("decoder_model.onnx"),
            encoder = encoderWrappers("encoder_model.onnx"))
        val spp = readSentencePieceModel(path, spark, "_nllb_spp", sppFile)
        instance.setModelIfNotSet(spark, onnxWrappers, spp)
      case _ =>
        throw new Exception(notSupportedEngineError)
    }
  }

  addReader(readModel)

  def loadSavedModel(modelPath: String, spark: SparkSession): NLLBTransformer = {
    implicit val formats: DefaultFormats.type = DefaultFormats // for json4
    val (localModelPath, detectedEngine) =
      modelSanityCheck(modelPath, isDecoder = true)
    val modelConfig: JValue =
      parse(loadJsonStringAsset(localModelPath, "config.json"))

    val beginSuppressTokens: Array[Int] =
      (modelConfig \ "begin_suppress_tokens").extract[Array[Int]]

    val suppressTokenIds: Array[Int] =
      (modelConfig \ "suppress_tokens").extract[Array[Int]]

    val forcedDecoderIds: Array[(Int, Int)] = Array()

    def arrayOrNone[T](array: Array[T]): Option[Array[T]] =
      if (array.nonEmpty) Some(array) else None

    val bosTokenId = (modelConfig \ "bos_token_id").extract[Int]
    val eosTokenId = (modelConfig \ "eos_token_id").extract[Int]
    val padTokenId = (modelConfig \ "eos_token_id").extract[Int]
    val vocabSize = (modelConfig \ "vocab_size").extract[Int]

    val annotatorModel = new NLLBTransformer()
      .setGenerationConfig(
        GenerationConfig(
          bosTokenId,
          padTokenId,
          eosTokenId,
          vocabSize,
          arrayOrNone(beginSuppressTokens),
          arrayOrNone(suppressTokenIds),
          arrayOrNone(forcedDecoderIds)))
    val spModel = loadSentencePieceAsset(localModelPath, "sentencepiece.bpe.model")
    val vocabulary: JValue =
      parse(loadJsonStringAsset(localModelPath, "vocab.json"))
    // convert to map
    val vocab = vocabulary.extract[Map[String, Int]]
    annotatorModel.setVocabulary(vocab)
    annotatorModel.set(annotatorModel.engine, detectedEngine)

    detectedEngine match {
      case ONNX.name =>
        val onnxWrapperEncoder =
          OnnxWrapper.read(
            localModelPath,
            zipped = false,
            useBundle = true,
            modelName = "encoder_model")
        val onnxWrapperDecoder =
          OnnxWrapper.read(
            localModelPath,
            zipped = false,
            useBundle = true,
            modelName = "decoder_model")

        val onnxWrappers =
          EncoderDecoderWithoutPastWrappers(
            encoder = onnxWrapperEncoder,
            decoder = onnxWrapperDecoder)

        annotatorModel
          .setModelIfNotSet(spark, onnxWrappers, spModel)

      case _ =>
        throw new Exception(notSupportedEngineError)
    }

    annotatorModel
  }

}

object NLLBTransformer
    extends ReadablePretrainedNLLBTransformerModel
    with ReadNLLBTransformerDLModel
