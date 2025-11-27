#  Copyright 2017-2024 John Snow Labs
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Contains classes for the NLLBTransformer."""

from sparknlp.common import *


class NLLBTransformer(AnnotatorModel, HasBatchedAnnotate, HasEngine):
    """NLLB : multilingual translation model

    NLLB is a multilingual encoder-decoder (seq-to-seq) model trained for Many-to-Many
    multilingual translation.

    The model can directly translate between 200+ languages.

    Pretrained models can be loaded with :meth:`.pretrained` of the companion
    object:

    >>> nllb = NLLBTransformer.pretrained() \\
    ...     .setInputCols(["document"]) \\
    ...     .setOutputCol("generation")


    The default model is ``"nllb_distilled_600M_8int"``, if no name is provided. For available
    pretrained models please see the `Models Hub
    <https://sparknlp.org/models?q=nllb>`__.

    ====================== ======================
    Input Annotation types Output Annotation type
    ====================== ======================
    ``DOCUMENT``           ``DOCUMENT``
    ====================== ======================

    Parameters
    ----------
    configProtoBytes
        ConfigProto from tensorflow, serialized into byte array.
    minOutputLength
        Minimum length of the sequence to be generated, by default 0
    maxOutputLength
        Maximum length of output text, by default 20
    doSample
        Whether or not to use sampling; use greedy decoding otherwise, by default False
    temperature
        The value used to module the next token probabilities, by default 1.0
    topK
        The number of highest probability vocabulary tokens to keep for
        top-k-filtering, by default 50
    topP
        Top cumulative probability for vocabulary tokens, by default 1.0

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.
    repetitionPenalty
        The parameter for repetition penalty, 1.0 means no penalty. , by default
        1.0
    noRepeatNgramSize
        If set to int > 0, all ngrams of that size can only occur once, by
        default 0
    ignoreTokenIds
        A list of token ids which are ignored in the decoder's output, by
        default []
    srcLang
        Source Language (Default: `en`)
    tgtLang
        Target Language (Default: `fr`)

    Languages Covered
    -----------------
    Acehnese (Arabic script) (ace_Arab), Acehnese (Latin script) (ace_Latn), Mesopotamian Arabic
    (acm_Arab), Ta’izzi-Adeni Arabic (acq_Arab), Tunisian Arabic (aeb_Arab), Afrikaans (afr_Latn),
    South Levantine Arabic (ajp_Arab), Akan (aka_Latn), Amharic (amh_Ethi), North Levantine Arabic
    (apc_Arab), Modern Standard Arabic (arb_Arab), Modern Standard Arabic (Romanized) (arb_Latn),
    Najdi Arabic (ars_Arab), Moroccan Arabic (ary_Arab), Egyptian Arabic (arz_Arab), Assamese
    (asm_Beng), Asturian (ast_Latn), Awadhi (awa_Deva), Central Aymara (ayr_Latn), South
    Azerbaijani (azb_Arab), North Azerbaijani (azj_Latn), Bashkir (bak_Cyrl), Bambara (bam_Latn),
    Balinese (ban_Latn), Belarusian (bel_Cyrl), Bemba (bem_Latn), Bengali (ben_Beng), Bhojpuri
    (bho_Deva), Banjar (Arabic script) (bjn_Arab), Banjar (Latin script) (bjn_Latn), Standard
    Tibetan (bod_Tibt), Bosnian (bos_Latn), Buginese (bug_Latn), Bulgarian (bul_Cyrl), Catalan
    (cat_Latn), Cebuano (ceb_Latn), Czech (ces_Latn), Chokwe (cjk_Latn), Central Kurdish
    (ckb_Arab), Crimean Tatar (crh_Latn), Welsh (cym_Latn), Danish (dan_Latn), German (deu_Latn),
    Southwestern Dinka (dik_Latn), Dyula (dyu_Latn), Dzongkha (dzo_Tibt), Greek (ell_Grek),
    English (eng_Latn), Esperanto (epo_Latn), Estonian (est_Latn), Basque (eus_Latn), Ewe
    (ewe_Latn), Faroese (fao_Latn), Fijian (fij_Latn), Finnish (fin_Latn), Fon (fon_Latn), French
    (fra_Latn), Friulian (fur_Latn), Nigerian Fulfulde (fuv_Latn), Scottish Gaelic (gla_Latn),
    Irish (gle_Latn), Galician (glg_Latn), Guarani (grn_Latn), Gujarati (guj_Gujr), Haitian Creole
    (hat_Latn), Hausa (hau_Latn), Hebrew (heb_Hebr), Hindi (hin_Deva), Chhattisgarhi (hne_Deva),
    Croatian (hrv_Latn), Hungarian (hun_Latn), Armenian (hye_Armn), Igbo (ibo_Latn), Ilocano
    (ilo_Latn), Indonesian (ind_Latn), Icelandic (isl_Latn), Italian (ita_Latn), Javanese
    (jav_Latn), Japanese (jpn_Jpan), Kabyle (kab_Latn), Jingpho (kac_Latn), Kamba (kam_Latn),
    Kannada (kan_Knda), Kashmiri (Arabic script) (kas_Arab), Kashmiri (Devanagari script)
    (kas_Deva), Georgian (kat_Geor), Central Kanuri (Arabic script) (knc_Arab), Central Kanuri
    (Latin script) (knc_Latn), Kazakh (kaz_Cyrl), Kabiyè (kbp_Latn), Kabuverdianu (kea_Latn),
    Khmer (khm_Khmr), Kikuyu (kik_Latn), Kinyarwanda (kin_Latn), Kyrgyz (kir_Cyrl), Kimbundu
    (kmb_Latn), Northern Kurdish (kmr_Latn), Kikongo (kon_Latn), Korean (kor_Hang), Lao
    (lao_Laoo), Ligurian (lij_Latn), Limburgish (lim_Latn), Lingala (lin_Latn), Lithuanian
    (lit_Latn), Lombard (lmo_Latn), Latgalian (ltg_Latn), Luxembourgish (ltz_Latn), Luba-Kasai
    (lua_Latn), Ganda (lug_Latn), Luo (luo_Latn), Mizo (lus_Latn), Standard Latvian (lvs_Latn),
    Magahi (mag_Deva), Maithili (mai_Deva), Malayalam (mal_Mlym), Marathi (mar_Deva), Minangkabau
    (Arabic script) (min_Arab), Minangkabau (Latin script) (min_Latn), Macedonian (mkd_Cyrl),
    Plateau Malagasy (plt_Latn), Maltese (mlt_Latn), Meitei (Bengali script) (mni_Beng), Halh
    Mongolian (khk_Cyrl), Mossi (mos_Latn), Maori (mri_Latn), Burmese (mya_Mymr), Dutch
    (nld_Latn), Norwegian Nynorsk (nno_Latn), Norwegian Bokmål (nob_Latn), Nepali (npi_Deva),
    Northern Sotho (nso_Latn), Nuer (nus_Latn), Nyanja (nya_Latn), Occitan (oci_Latn), West
    Central Oromo (gaz_Latn), Odia (ory_Orya), Pangasinan (pag_Latn), Eastern Panjabi (pan_Guru),
    Papiamento (pap_Latn), Western Persian (pes_Arab), Polish (pol_Latn), Portuguese (por_Latn),
    Dari (prs_Arab), Southern Pashto (pbt_Arab), Ayacucho Quechua (quy_Latn), Romanian (ron_Latn),
    Rundi (run_Latn), Russian (rus_Cyrl), Sango (sag_Latn), Sanskrit (san_Deva), Santali
    (sat_Olck), Sicilian (scn_Latn), Shan (shn_Mymr), Sinhala (sin_Sinh), Slovak (slk_Latn),
    Slovenian (slv_Latn), Samoan (smo_Latn), Shona (sna_Latn), Sindhi (snd_Arab), Somali
    (som_Latn), Southern Sotho (sot_Latn), Spanish (spa_Latn), Tosk Albanian (als_Latn), Sardinian
    (srd_Latn), Serbian (srp_Cyrl), Swati (ssw_Latn), Sundanese (sun_Latn), Swedish (swe_Latn),
    Swahili (swh_Latn), Silesian (szl_Latn), Tamil (tam_Taml), Tatar (tat_Cyrl), Telugu
    (tel_Telu), Tajik (tgk_Cyrl), Tagalog (tgl_Latn), Thai (tha_Thai), Tigrinya (tir_Ethi),
    Tamasheq (Latin script) (taq_Latn), Tamasheq (Tifinagh script) (taq_Tfng), Tok Pisin
    (tpi_Latn), Tswana (tsn_Latn), Tsonga (tso_Latn), Turkmen (tuk_Latn), Tumbuka (tum_Latn),
    Turkish (tur_Latn), Twi (twi_Latn), Central Atlas Tamazight (tzm_Tfng), Uyghur (uig_Arab),
    Ukrainian (ukr_Cyrl), Umbundu (umb_Latn), Urdu (urd_Arab), Northern Uzbek (uzn_Latn), Venetian
    (vec_Latn), Vietnamese (vie_Latn), Waray (war_Latn), Wolof (wol_Latn), Xhosa (xho_Latn),
    Eastern Yiddish (ydd_Hebr), Yoruba (yor_Latn), Yue Chinese (yue_Hant), Chinese (Simplified)
    (zho_Hans), Chinese (Traditional) (zho_Hant), Standard Malay (zsm_Latn), Zulu (zul_Latn).
   

    References
    ----------
    - `Beyond English-Centric Multilingual Machine Translation
      <https://arxiv.org/pdf/2010.11125.pdf>`__
    - https://github.com/pytorch/fairseq/tree/master/examples/m2m_100

    **Paper Abstract:**

    *Driven by the goal of eradicating language barriers on a global scale, machine translation has solidified itself
    as a key focus of artificial intelligence research today. However, such efforts have coalesced around a small
    subset of languages, leaving behind the vast majority of mostly low-resource languages. What does it take to
    break the 200 language barrier while ensuring safe, high quality results, all while keeping ethical
    considerations in mind? In No Language Left Behind, we took on this challenge by first contextualizing the need
    for low-resource language translation support through exploratory interviews with native speakers. Then,
    we created datasets and models aimed at narrowing the performance gap between low and high-resource languages.
    More specifically, we developed a conditional compute model based on Sparsely Gated Mixture of Experts that is
    trained on data obtained with novel and effective data mining techniques tailored for low-resource languages. We
    propose multiple architectural and training improvements to counteract overfitting while training on thousands of
    tasks. Critically, we evaluated the performance of over 40,000 different translation directions using a
    human-translated benchmark, Flores-200, and combined human evaluation with a novel toxicity benchmark covering
    all languages in Flores-200 to assess translation safety. Our model achieves an improvement of 44% BLEU relative
    to the previous state-of-the-art, laying important groundwork towards realizing a universal translation system.*

    Examples
    --------
    >>> import sparknlp
    >>> from sparknlp.base import *
    >>> from sparknlp.annotator import *
    >>> from pyspark.ml import Pipeline
    >>> documentAssembler = DocumentAssembler() \\
    ...     .setInputCol("text") \\
    ...     .setOutputCol("documents")
    >>> nllb = NLLBTransformer.pretrained("nllb_distilled_600M_8int") \\
    ...     .setInputCols(["documents"]) \\
    ...     .setMaxOutputLength(50) \\
    ...     .setOutputCol("generation") \\
    ...     .setSrcLang("zho_Hans") \\
    ...     .setTgtLang("eng_Latn")
    >>> pipeline = Pipeline().setStages([documentAssembler, nllb])
    >>> data = spark.createDataFrame([["生活就像一盒巧克力。"]]).toDF("text")
    >>> result = pipeline.fit(data).transform(data)
    >>> result.select("summaries.generation").show(truncate=False)
    +-------------------------------------------------------------------------------------------+
    |result                                                                                     |
    +-------------------------------------------------------------------------------------------+
    |[ Life is like a box of chocolate.]                                                        |
    +-------------------------------------------------------------------------------------------+
    """

    name = "NLLBTransformer"

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT]

    outputAnnotatorType = AnnotatorType.DOCUMENT

    configProtoBytes = Param(Params._dummy(), "configProtoBytes",
                             "ConfigProto from tensorflow, serialized into byte array. Get with config_proto.SerializeToString()",
                             TypeConverters.toListInt)

    minOutputLength = Param(Params._dummy(), "minOutputLength", "Minimum length of the sequence to be generated",
                            typeConverter=TypeConverters.toInt)

    maxOutputLength = Param(Params._dummy(), "maxOutputLength", "Maximum length of output text",
                            typeConverter=TypeConverters.toInt)

    doSample = Param(Params._dummy(), "doSample", "Whether or not to use sampling; use greedy decoding otherwise",
                     typeConverter=TypeConverters.toBoolean)

    temperature = Param(Params._dummy(), "temperature", "The value used to module the next token probabilities",
                        typeConverter=TypeConverters.toFloat)

    topK = Param(Params._dummy(), "topK",
                 "The number of highest probability vocabulary tokens to keep for top-k-filtering",
                 typeConverter=TypeConverters.toInt)

    topP = Param(Params._dummy(), "topP",
                 "If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation",
                 typeConverter=TypeConverters.toFloat)

    repetitionPenalty = Param(Params._dummy(), "repetitionPenalty",
                              "The parameter for repetition penalty. 1.0 means no penalty. See `this paper <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details",
                              typeConverter=TypeConverters.toFloat)

    noRepeatNgramSize = Param(Params._dummy(), "noRepeatNgramSize",
                              "If set to int > 0, all ngrams of that size can only occur once",
                              typeConverter=TypeConverters.toInt)

    ignoreTokenIds = Param(Params._dummy(), "ignoreTokenIds",
                           "A list of token ids which are ignored in the decoder's output",
                           typeConverter=TypeConverters.toListInt)
    beamSize = Param(Params._dummy(), "beamSize", "The Number of beams for beam search.",
                     typeConverter=TypeConverters.toInt)
    srcLang = Param(Params._dummy(), "srcLang", "Source Language (Default: `en`)",
                    typeConverter=TypeConverters.toString)
    tgtLang = Param(Params._dummy(), "tgtLang", "Target Language (Default: `fr`)",
                    typeConverter=TypeConverters.toString)

    def setIgnoreTokenIds(self, value):
        """A list of token ids which are ignored in the decoder's output.

        Parameters
        ----------
        value : List[int]
            The words to be filtered out
        """
        return self._set(ignoreTokenIds=value)

    def setConfigProtoBytes(self, b):
        """Sets configProto from tensorflow, serialized into byte array.

        Parameters
        ----------
        b : List[int]
            ConfigProto from tensorflow, serialized into byte array
        """
        return self._set(configProtoBytes=b)

    def setMinOutputLength(self, value):
        """Sets minimum length of the sequence to be generated.

        Parameters
        ----------
        value : int
            Minimum length of the sequence to be generated
        """
        return self._set(minOutputLength=value)

    def setMaxOutputLength(self, value):
        """Sets maximum length of output text.

        Parameters
        ----------
        value : int
            Maximum length of output text
        """
        return self._set(maxOutputLength=value)

    def setDoSample(self, value):
        """Sets whether or not to use sampling, use greedy decoding otherwise.

        Parameters
        ----------
        value : bool
            Whether or not to use sampling; use greedy decoding otherwise
        """
        return self._set(doSample=value)

    def setTemperature(self, value):
        """Sets the value used to module the next token probabilities.

        Parameters
        ----------
        value : float
            The value used to module the next token probabilities
        """
        return self._set(temperature=value)

    def setTopK(self, value):
        """Sets the number of highest probability vocabulary tokens to keep for
        top-k-filtering.

        Parameters
        ----------
        value : int
            Number of highest probability vocabulary tokens to keep
        """
        return self._set(topK=value)

    def setTopP(self, value):
        """Sets the top cumulative probability for vocabulary tokens.

        If set to float < 1, only the most probable tokens with probabilities
        that add up to ``topP`` or higher are kept for generation.

        Parameters
        ----------
        value : float
            Cumulative probability for vocabulary tokens
        """
        return self._set(topP=value)

    def setRepetitionPenalty(self, value):
        """Sets the parameter for repetition penalty. 1.0 means no penalty.

        Parameters
        ----------
        value : float
            The repetition penalty

        References
        ----------
        See `Ctrl: A Conditional Transformer Language Model For Controllable
        Generation <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
        """
        return self._set(repetitionPenalty=value)

    def setNoRepeatNgramSize(self, value):
        """Sets size of n-grams that can only occur once.

        If set to int > 0, all ngrams of that size can only occur once.

        Parameters
        ----------
        value : int
            N-gram size can only occur once
        """
        return self._set(noRepeatNgramSize=value)

    def setBeamSize(self, value):
        """Sets the number of beam size for beam search, by default `4`.

        Parameters
        ----------
        value : int
            Number of beam size for beam search
        """
        return self._set(beamSize=value)

    def setSrcLang(self, value):
        """Sets source language.

        Parameters
        ----------
        value : str
            Source language
        """
        return self._set(srcLang=value)

    def setTgtLang(self, value):
        """Sets target language.

        Parameters
        ----------
        value : str
            Target language
        """
        return self._set(tgtLang=value)

    @keyword_only
    def __init__(self, classname="com.johnsnowlabs.nlp.annotators.seq2seq.NLLBTransformer", java_model=None):
        super(NLLBTransformer, self).__init__(classname=classname, java_model=java_model)
        self._setDefault(minOutputLength=0, maxOutputLength=200, doSample=False, temperature=1, topK=50, topP=1,
                         repetitionPenalty=1.0, noRepeatNgramSize=0, ignoreTokenIds=[], batchSize=1, beamSize=1,
                         srcLang="en", tgtLang="fr")

    @staticmethod
    def loadSavedModel(folder, spark_session,  use_openvino=False):
        """Loads a locally saved model.

        Parameters
        ----------
        folder : str
            Folder of the saved model
        spark_session : pyspark.sql.SparkSession
            The current SparkSession

        Returns
        -------
        NLLBTransformer
            The restored model
        """
        from sparknlp.internal import _NLLBLoader
        jModel = _NLLBLoader(folder, spark_session._jsparkSession, use_openvino)._java_obj
        return NLLBTransformer(java_model=jModel)

    @staticmethod
    def pretrained(name="nllb_distilled_600M_8int", lang="xx", remote_loc=None):
        """Downloads and loads a pretrained model.

        Parameters
        ----------
        name : str, optional
            Name of the pretrained model, by default "nllb_distilled_600M_8int"
        lang : str, optional
            Language of the pretrained model, by default "en"
        remote_loc : str, optional
            Optional remote address of the resource, by default None. Will use
            Spark NLPs repositories otherwise.

        Returns
        -------
        NLLBTransformer
            The restored model
        """
        from sparknlp.pretrained import ResourceDownloader
        return ResourceDownloader.downloadModel(NLLBTransformer, name, lang, remote_loc)
