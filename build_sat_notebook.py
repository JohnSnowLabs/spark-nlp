# -*- coding: utf-8 -*-
"""Builds the SentenceDetectorSaT example notebook with correct JSON escaping."""
import json

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

# ----------------------------------------------------------------------------
# Long multilingual passages (each >= 1200 characters, random everyday topics).
# No expected counts are given anywhere — we just let SaT segment them.
# ----------------------------------------------------------------------------
EN = (
    "Modern smartphones have quietly become the most important computing devices in human history. "
    "A typical flagship released in 2024 ships with a 6.7 inch OLED display, a chipset built on a 3 nm process, and at least 12 GB of RAM. "
    "Photography is where the marketing battles are fiercest. "
    "Manufacturers now advertise 200 MP main sensors, periscope zoom lenses, and computational night modes that stack dozens of frames in under a second. "
    "Battery technology, by contrast, has improved far more slowly. "
    "Most phones still rely on lithium ion cells rated around 5,000 mAh, although charging speeds have leapt forward dramatically. "
    "Some brands now claim a full charge in roughly twenty minutes. "
    "Software support has also become a genuine differentiator. "
    "A few vendors promise up to seven years of operating system updates, which would have been unthinkable a decade ago. "
    "Repairability, however, remains a sore point for consumers and regulators alike. "
    "Glued batteries, fragile glass backs, and proprietary screws make do it yourself fixes difficult. "
    "The European Union has responded with right to repair rules and a mandate for USB C charging. "
    "Whether these regulations meaningfully change buyer behaviour is still an open question. "
    "For now, the average person upgrades roughly every three years, often nudged along by a carrier contract rather than a failing device."
)

DE = (
    "Ein stabiles WLAN gehört heute in fast jedem Haushalt zur Grundausstattung. "
    "Trotzdem klagen viele Menschen über langsame Verbindungen, abbrechende Videoanrufe und tote Funklöcher im Schlafzimmer. "
    "Die Ursachen sind oft erstaunlich banal. "
    "Dicke Betonwände, alte Router und überfüllte Funkkanäle bremsen das Signal spürbar aus. "
    "Der moderne Standard Wi Fi 6 verspricht höhere Geschwindigkeiten und eine bessere Verwaltung vieler Geräte gleichzeitig. "
    "In einer typischen Wohnung sind schnell zwanzig oder dreißig Geräte gleichzeitig verbunden. "
    "Smartphones, Fernseher, Lautsprecher und sogar Glühbirnen wollen alle ins Netz. "
    "Experten empfehlen daher, den Router möglichst zentral und frei aufzustellen. "
    "Ein Standort hinter dem Fernseher oder im Schrank ist fast immer eine schlechte Idee. "
    "Wer große Flächen abdecken muss, greift zunehmend zu sogenannten Mesh Systemen. "
    "Mehrere kleine Stationen spannen dabei ein einziges, nahtloses Netz auf. "
    "Auch die Sicherheit wird häufig vernachlässigt. "
    "Viele Nutzer ändern das voreingestellte Passwort nie und riskieren damit unbefugte Zugriffe. "
    "Regelmäßige Firmware Updates schließen zudem bekannte Sicherheitslücken und sollten nicht ignoriert werden. "
    "Ein gutes Heimnetz ist am Ende eine Mischung aus richtiger Technik, klugem Standort und etwas Geduld beim Einrichten."
)

PT = (
    "A Copa do Mundo da FIFA é, sem dúvida, o maior evento esportivo do planeta. "
    "A cada quatro anos, bilhões de pessoas param para acompanhar trinta e duas seleções, em breve quarenta e oito, disputando o título mais cobiçado do futebol. "
    "O Brasil é o país mais vitorioso da história da competição, com cinco conquistas. "
    "A primeira edição aconteceu em 1930, no Uruguai, e reuniu apenas treze equipes. "
    "Desde então, o torneio cresceu de forma impressionante. "
    "Hoje, os direitos de transmissão movimentam quantias bilionárias e transformam jogadores em celebridades globais. "
    "A tecnologia também mudou o jogo profundamente. "
    "O árbitro de vídeo, conhecido como VAR, passou a revisar lances polêmicos em poucos segundos. "
    "Muitos torcedores adoram a precisão, enquanto outros reclamam que a emoção espontânea se perdeu. "
    "Fora de campo, a escolha das sedes gera debates intensos. "
    "Questões como custos, sustentabilidade e direitos humanos aparecem em quase todas as edições. "
    "Apesar das polêmicas, a paixão popular permanece intacta. "
    "Quando a bola rola na final, diferenças políticas e culturais parecem desaparecer por noventa minutos. "
    "Crianças em todo o mundo passam a sonhar com a camisa da seleção do seu país. "
    "É esse poder de unir pessoas tão diferentes que torna a Copa verdadeiramente única."
)

NL = (
    "Koffie is in Nederland veel meer dan zomaar een drankje. "
    "Voor veel mensen begint de dag pas echt na het eerste kopje. "
    "Op kantoor is het koffieapparaat vaak de plek waar collega's elkaar even spreken. "
    "De gemiddelde Nederlander drinkt meerdere koppen per dag, meestal gewoon zwart of met een wolkje melk. "
    "De laatste jaren is er echter veel veranderd. "
    "Speciale koffiebars schieten in de grote steden als paddenstoelen uit de grond. "
    "Barista's praten met trots over bonen, branding en de juiste temperatuur van het water. "
    "Een cappuccino kost in zo'n zaak al snel meer dan vier euro. "
    "Toch blijven veel mensen trouw aan hun vertrouwde filterkoffie thuis. "
    "Duurzaamheid speelt een steeds grotere rol bij de keuze. "
    "Consumenten letten op keurmerken, eerlijke handel en herbruikbare bekers. "
    "Ook thuis investeren mensen in dure machines en verse bonen. "
    "De simpele oploskoffie raakt langzaam uit de gratie. "
    "Sommige mensen zweren bij een ouderwetse percolator, terwijl anderen niet zonder hun moderne espressomachine kunnen. "
    "De geur van versgemalen bonen in de ochtend is voor velen gewoon onmisbaar. "
    "Cafeïne blijft natuurlijk de belangrijkste reden om door te drinken. "
    "Wat ooit een snelle gewoonte was, is voor sommigen bijna een hobby geworden. "
    "Of je nu kiest voor gemak of voor smaak, koffie blijft een vast onderdeel van het dagelijks leven."
)

TR = (
    "Çay, Türkiye'de günlük yaşamın vazgeçilmez bir parçasıdır. "
    "Sabah kahvaltısından gece sohbetlerine kadar neredeyse her ana eşlik eder. "
    "İnce belli bardakta servis edilen koyu renkli çay, misafirperverliğin de bir simgesidir. "
    "Bir eve konuk gittiğinizde, size mutlaka taze demlenmiş bir bardak çay ikram edilir. "
    "Türkiye, dünyada kişi başına en çok çay tüketen ülkelerden biridir. "
    "Çayın büyük bölümü Karadeniz kıyısındaki Rize ve çevresinde yetiştirilir. "
    "Geleneksel demleme yöntemi, çaydanlık adı verilen iki katlı bir kapla yapılır. "
    "Alt kısımda su kaynarken, üst kısımda çay yavaşça demlenir. "
    "İyi bir çayın rengi parlak, tadı ise ne fazla acı ne de fazla açık olmalıdır. "
    "Şehirlerde çay bahçeleri, arkadaşların buluştuğu popüler mekanlardır. "
    "İnsanlar burada saatlerce oturup sohbet eder, oyun oynar ve dinlenir. "
    "Son yıllarda kahve kültürü de hızla yayılıyor. "
    "Bazı bölgelerde çayın yanında lokum veya küçük tatlılar da ikram edilir. "
    "Kış aylarında sıcak bir bardak çay, insana adeta huzur verir. "
    "Kimi zaman çaya nane ya da adaçayı gibi bitkiler de eklenir. "
    "Yine de çay, kültürel kimliğin merkezindeki yerini korumaya devam ediyor. "
    "Kısacası bir bardak çay, çoğu zaman sadece bir içecek değil, bir buluşma bahanesidir."
)

ID = (
    "Sepeda motor listrik semakin populer di kota-kota besar Indonesia. "
    "Harga bahan bakar yang terus naik membuat banyak orang mencari alternatif yang lebih murah. "
    "Selain hemat, kendaraan listrik juga dianggap lebih ramah lingkungan. "
    "Pemerintah pun mulai memberikan subsidi untuk mendorong masyarakat beralih. "
    "Namun, tantangannya masih cukup besar. "
    "Jumlah stasiun pengisian daya umum masih sangat terbatas di banyak daerah. "
    "Banyak calon pembeli khawatir kehabisan baterai di tengah jalan. "
    "Jarak tempuh sekali pengisian biasanya berkisar antara enam puluh hingga seratus kilometer. "
    "Untuk perjalanan dalam kota, angka itu sebenarnya sudah cukup memadai. "
    "Bengkel khusus motor listrik juga belum tersebar secara merata. "
    "Akibatnya, sebagian orang masih ragu soal perawatan jangka panjang. "
    "Meski begitu, minat masyarakat terus tumbuh dari tahun ke tahun. "
    "Beberapa perusahaan lokal bahkan mulai memproduksi model mereka sendiri. "
    "Mereka menawarkan harga yang lebih terjangkau dibandingkan merek impor. "
    "Jika infrastruktur terus diperbaiki, masa depan kendaraan listrik di Indonesia tampak cerah. "
    "Dengan dukungan kebijakan yang tepat, transisi ini bisa berjalan jauh lebih cepat. "
    "Perubahan kebiasaan memang butuh waktu, tetapi arah tujuannya sudah semakin jelas."
)

PASSAGES = [("English", EN), ("German", DE), ("Portuguese", PT),
            ("Dutch", NL), ("Turkish", TR), ("Indonesian", ID)]

# Enforce the >= 1200 character requirement.
for name, txt in PASSAGES:
    assert len(txt) >= 1200, f"{name} is only {len(txt)} chars (<1200)"
    print(f"{name:12s} {len(txt)} chars")

# Build the python literal for the samples list, escaped safely via json.dumps.
samples_lines = ",\n".join(
    f'    ("{name}", {json.dumps(txt, ensure_ascii=False)})' for name, txt in PASSAGES
)
samples_block = "samples = [\n" + samples_lines + ",\n]"

# ----------------------------------------------------------------------------
# Notebook cells
# ----------------------------------------------------------------------------
cells = []

cells.append(md(
    "![JohnSnowLabs](https://sparknlp.org/assets/images/logo.png)\n\n"
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/JohnSnowLabs/spark-nlp/blob/master/"
    "examples/python/transformers/onnx/HuggingFace_ONNX_in_Spark_NLP_SentenceDetectorSaT.ipynb)"
))

cells.append(md(
    "## Import SaT (Segment any Text) sentence detector from HuggingFace 🤗 into Spark NLP 🚀\n\n"
    "This notebook introduces **`SentenceDetectorSaTModel`** — a new, transformer-based sentence boundary "
    "detector in Spark NLP built on the [wtpsplit / **S**egment **a**ny **T**ext](https://github.com/segment-any-text/wtpsplit) models.\n\n"
    "Unlike the classic rule-based `SentenceDetector` (which splits on punctuation patterns) or "
    "`SentenceDetectorDLModel` (a CNN trained mostly on European-language corpora), SaT predicts a boundary "
    "probability for **every sub-word token** with an XLM-RoBERTa backbone. In practice this means it:\n\n"
    "- **doesn't blindly split on punctuation** — it works on noisy, lower-cased, or completely unpunctuated "
    "text (chat logs, ASR transcripts, OCR), where rule-based detectors fail;\n"
    "- **generalizes across 85+ languages** out of a single model;\n"
    "- **adjusts the boundary to the best spot** instead of cutting at a fixed offset, with an optional "
    "**minimum / maximum sentence length** (length-constrained Viterbi search).\n\n"
    "A few things to keep in mind before we start 😊\n\n"
    "- ONNX support was introduced in `Spark NLP 5.0.0` for high-performance inference.\n"
    "- `SentenceDetectorSaTModel` is available in the Spark NLP release that introduced SaT (**6.5.0+**).\n"
    "- The default model is [`segment-any-text/sat-12l-sm`](https://huggingface.co/segment-any-text/sat-12l-sm): "
    "12 Transformer layers, **85+ languages**, MIT licensed."
))

cells.append(md(
    "## 1. Export the model to ONNX (fp32)\n\n"
    "The SaT repo *does* publish ONNX files (`model.onnx`, `model_optimized.onnx`), but they are exported in "
    "**fp16** — that variant is GPU-oriented and is awkward to run on CPU, which is the most common Spark NLP "
    "deployment. So instead of downloading those, we re-export the model ourselves in **fp32** straight from the "
    "safetensors weights. This is the reliable path that avoids the import/runtime issues you can hit with the "
    "prebuilt fp16 graph.\n\n"
    "Two details matter for SaT specifically:\n\n"
    "- the architecture is a custom `xlm-token` (SubwordXLM) head, so we `import wtpsplit.models` to register it "
    "with `transformers` before calling `AutoModelForTokenClassification`;\n"
    "- the ONNX I/O contract is `input_ids` (int64) + **`attention_mask` as a float tensor** -> `logits`. "
    "The float mask is how wtpsplit exports it, and Spark NLP expects the same."
))

cells.append(code(
    '!pip install -q "transformers>=4.30" torch onnx onnxruntime sentencepiece wtpsplit'
))

cells.append(code(
    "from pathlib import Path\n"
    "import torch\n"
    "from transformers import AutoModelForTokenClassification, AutoTokenizer\n"
    "import wtpsplit.models  # noqa: F401 -> registers the SubwordXLM ('xlm-token') architecture\n\n"
    'HF_MODEL_ID  = "segment-any-text/sat-12l-sm"\n'
    'TOKENIZER_ID = "FacebookAI/xlm-roberta-base"   # SaT reuses the XLM-R base SentencePiece tokenizer\n'
    'EXPORT_DIR   = "sat-12l-sm-onnx"\n'
    'ASSETS_DIR   = f"{EXPORT_DIR}/assets"\n'
    "Path(ASSETS_DIR).mkdir(parents=True, exist_ok=True)\n\n"
    "# 1) Load the model and export it to fp32 ONNX with a FLOAT attention mask + dynamic axes\n"
    "model = AutoModelForTokenClassification.from_pretrained(HF_MODEL_ID).eval()\n\n"
    "dummy_ids  = torch.randint(0, model.config.vocab_size, (1, 16), dtype=torch.int64)\n"
    "dummy_mask = torch.ones((1, 16), dtype=torch.float32)   # SaT exports attention_mask as FLOAT\n\n"
    "torch.onnx.export(\n"
    "    model,\n"
    '    ({"input_ids": dummy_ids, "attention_mask": dummy_mask},),\n'
    '    f"{EXPORT_DIR}/model.onnx",\n'
    '    input_names=["input_ids", "attention_mask"],\n'
    '    output_names=["logits"],\n'
    "    dynamic_axes={\n"
    '        "input_ids":      {0: "batch", 1: "sequence"},\n'
    '        "attention_mask": {0: "batch", 1: "sequence"},\n'
    '        "logits":         {0: "batch", 1: "sequence"},\n'
    "    },\n"
    "    opset_version=14,\n"
    "    do_constant_folding=True,\n"
    ")\n\n"
    "# 2) Save the tokenizer; Spark NLP reads sentencepiece.bpe.model from the assets/ folder\n"
    "AutoTokenizer.from_pretrained(TOKENIZER_ID).save_pretrained(ASSETS_DIR)\n"
    'print("exported ->", EXPORT_DIR)'
))

cells.append(md(
    "Let's confirm the folder layout. Spark NLP needs `model.onnx` at the root and "
    "`assets/sentencepiece.bpe.model` (the other tokenizer files are harmless extras)."
))

cells.append(code("!ls -lR {EXPORT_DIR}"))

cells.append(md(
    "Quick sanity check on the exported graph — the inputs should be `input_ids` and a **float** "
    "`attention_mask`, and the single output should be `logits`."
))

cells.append(code(
    "import onnxruntime as ort\n\n"
    'sess = ort.InferenceSession(f"{EXPORT_DIR}/model.onnx", providers=["CPUExecutionProvider"])\n'
    'print("inputs :", [(i.name, i.type) for i in sess.get_inputs()])\n'
    'print("outputs:", [(o.name, o.type) for o in sess.get_outputs()])'
))

cells.append(md("## 2. Set up Spark NLP\n\nThis part is easy via our simple Colab script."))

cells.append(code("! wget -q http://setup.johnsnowlabs.com/colab.sh -O - | bash"))

cells.append(code(
    "import sparknlp\n\n"
    "# let's start Spark with Spark NLP\n"
    "spark = sparknlp.start()\n\n"
    'print("Apache Spark version: {}".format(spark.version))\n'
    'print("Spark NLP version:    {}".format(sparknlp.version()))'
))

cells.append(md(
    "## 3. Load the model with `loadSavedModel`\n\n"
    "`loadSavedModel` takes the export folder and the active `SparkSession`. The key SaT parameter is "
    "`threshold`: a boundary is placed once a character's predicted probability is `>= threshold`. "
    "The default `0.25` is calibrated for `sat-12l-sm`."
))

cells.append(code(
    "from sparknlp.annotator import *\n"
    "from sparknlp.base import *\n\n"
    "sat = SentenceDetectorSaTModel.loadSavedModel(EXPORT_DIR, spark) \\\n"
    '    .setInputCols(["document"]) \\\n'
    '    .setOutputCol("sentence") \\\n'
    "    .setThreshold(0.25)"
))

cells.append(md("Let's save it to disk so it can be moved around and reused later via `.load()`."))

cells.append(code('sat.write().overwrite().save("./sat_12l_sm_spark_nlp_onnx")'))

cells.append(md("Let's clean up the raw export — we don't need it anymore."))

cells.append(code("!rm -rf {EXPORT_DIR}"))

cells.append(md("Now we can reload the saved model anywhere — on another machine, cluster, or session 😊"))

cells.append(code(
    'sat_loaded = SentenceDetectorSaTModel.load("./sat_12l_sm_spark_nlp_onnx") \\\n'
    '    .setInputCols(["document"]) \\\n'
    '    .setOutputCol("sentence")'
))

cells.append(md(
    "> **Tip:** once a SaT model is on the [Spark NLP Models Hub](https://sparknlp.org/models), you can skip the "
    "whole import and just call the one-liner:\n>\n"
    "> ```python\n"
    '> sat = SentenceDetectorSaTModel.pretrained("sat_12l_sm", "xx") \\\n'
    '>     .setInputCols(["document"]).setOutputCol("sentence")\n'
    "> ```"
))

cells.append(md(
    "## 4. Quick start — basic sentence segmentation\n\n"
    "The input is a `DOCUMENT` column from `DocumentAssembler`, and the output is one `DOCUMENT` annotation "
    "per detected sentence."
))

cells.append(code(
    "from pyspark.ml import Pipeline\n\n"
    "document_assembler = DocumentAssembler() \\\n"
    '    .setInputCol("text") \\\n'
    '    .setOutputCol("document")\n\n'
    "pipeline = Pipeline(stages=[document_assembler, sat_loaded])\n\n"
    "data = spark.createDataFrame(\n"
    '    [["Dr. Smith flew to Washington D.C. on Monday. He met Mr. Brown at 5 p.m. to discuss the merger."]]\n'
    ').toDF("text")\n\n'
    "result = pipeline.fit(data).transform(data)\n"
    'result.selectExpr("explode(sentence.result) as sentence").show(truncate=False)'
))

cells.append(md(
    "## 5. SaT vs. the classic detectors — on messy text\n\n"
    "Spark NLP already ships two sentence detectors:\n\n"
    "- **`SentenceDetector`** — rule-based, splits on punctuation and casing patterns.\n"
    "- **`SentenceDetectorDLModel`** — a CNN boundary classifier; the multilingual `xx` model covers ~18 "
    "(mostly European) languages.\n\n"
    "Both do well on clean prose, but real-world text — chat, voice transcripts, OCR — is often "
    "**lower-cased and unpunctuated**, and they then return the whole blob as one \"sentence\". SaT predicts "
    "boundaries from the *language itself*. Let's run all three on the same input."
))

cells.append(code(
    "noisy_text = (\n"
    '    "so i went to the store yesterday and grabbed some milk then i realized "\n'
    '    "i forgot my wallet at home what a disaster i had to drive all the way back "\n'
    '    "by the time i returned the rain had started so i just stayed in and ordered pizza"\n'
    ")\n\n"
    "# Three detectors, all reading the same 'document' column, each writing its own output column\n"
    "rule_detector = SentenceDetector() \\\n"
    '    .setInputCols(["document"]) \\\n'
    '    .setOutputCol("rule_sentences")\n\n'
    'dl_detector = SentenceDetectorDLModel.pretrained("sentence_detector_dl", "xx") \\\n'
    '    .setInputCols(["document"]) \\\n'
    '    .setOutputCol("dl_sentences")\n\n'
    "sat_detector = sat_loaded \\\n"
    '    .setInputCols(["document"]) \\\n'
    '    .setOutputCol("sat_sentences")\n\n'
    "compare_pipeline = Pipeline(stages=[document_assembler, rule_detector, dl_detector, sat_detector])\n\n"
    'data = spark.createDataFrame([[noisy_text]]).toDF("text")\n'
    "result = compare_pipeline.fit(data).transform(data)\n\n"
    'for col_name, label in [("rule_sentences", "Rule-based SentenceDetector"),\n'
    '                        ("dl_sentences",   "SentenceDetectorDLModel (xx)"),\n'
    '                        ("sat_sentences",  "SentenceDetectorSaTModel")]:\n'
    '    sents = [r.s for r in result.selectExpr(f"explode({col_name}.result) as s").collect()]\n'
    '    print(f"{label} -> {len(sents)} sentence(s):")\n'
    "    for s in sents:\n"
    '        print("   •", s)\n'
    "    print()"
))

cells.append(md(
    "The rule-based detector finds **no boundaries** (no terminal punctuation) and returns one long run-on "
    "sentence; the DL model also leans on punctuation cues and recovers few, if any. SaT reconstructs the "
    "individual sentences from the wording alone — the robustness you need for transcripts and user-generated text."
))

cells.append(md(
    "## 6. Long-form multilingual comparison\n\n"
    "Now let's stress all three detectors on **long, real-world paragraphs (1200+ characters each) across six "
    "languages** — including less widely served ones like Portuguese, Dutch, Turkish, and Indonesian, all of "
    "which `sat-12l-sm` supports out of the box. The topics are deliberately ordinary (smartphones, home Wi-Fi, "
    "the World Cup, coffee, tea, electric scooters).\n\n"
    "We won't prescribe how many sentences each *should* produce — just run them and count what each detector "
    "finds. We reuse the exact `compare_pipeline` from the previous step, so it's apples-to-apples."
))

cells.append(code(
    "from pyspark.sql.functions import size\n\n"
    + samples_block + "\n\n"
    'lang_df = spark.createDataFrame(samples, ["lang", "text"])\n'
    "counted = compare_pipeline.fit(lang_df).transform(lang_df)\n\n"
    "counted.select(\n"
    '    "lang",\n'
    '    size("rule_sentences.result").alias("Rule"),\n'
    '    size("dl_sentences.result").alias("DL_xx"),\n'
    '    size("sat_sentences.result").alias("SaT"),\n'
    ").show(truncate=False)"
))

cells.append(md(
    "Across every language SaT segments the paragraph into a healthy set of well-formed sentences from a single "
    "model. The rule-based detector drifts — over-splitting on abbreviations and numbers, and under-splitting "
    "where punctuation cues are weaker — while the DL model's quality falls off on languages outside its training "
    "set. Let's actually look at SaT's splits for a couple of the languages."
))

cells.append(code(
    'for lang in ["English", "Turkish"]:\n'
    '    print(f"==== {lang} — SaT sentences ====")\n'
    "    rows = counted.where(f\"lang = '{lang}'\") \\\n"
    '        .selectExpr("explode(sat_sentences.result) as s").collect()\n'
    "    for i, r in enumerate(rows, 1):\n"
    '        print(f"{i:2d}. {r.s}")\n'
    '    print()'
))

cells.append(md(
    "## 7. Controlling sentence length — with boundary *adjustment*, not blind cutting\n\n"
    "Sometimes you need to cap (or floor) sentence length — e.g. to fit a downstream model's context window. "
    "The classic detector's `splitLength` **forcibly** cuts at a fixed offset regardless of meaning.\n\n"
    "SaT instead does a **length-constrained Viterbi search**: set `minSentenceLength` and/or `maxSentenceLength` "
    "(in characters) and it finds the globally highest-probability set of boundaries such that *every* segment "
    "respects the bounds. When a length cap is active, `threshold` is ignored. The result is segments that stay "
    "within your limits **and** land on the most natural boundary available."
))

cells.append(code(
    "sat_bounded = sat_loaded \\\n"
    '    .setInputCols(["document"]) \\\n'
    '    .setOutputCol("sentence") \\\n'
    "    .setMinSentenceLength(80) \\\n"
    "    .setMaxSentenceLength(160)\n\n"
    "pipeline = Pipeline(stages=[document_assembler, sat_bounded])\n"
    'data = spark.createDataFrame([[EN_TEXT]]).toDF("text")\n'
    "result = pipeline.fit(data).transform(data)\n\n"
    'sentences = [r.s for r in result.selectExpr("explode(sentence.result) as s").collect()]\n'
    "for i, s in enumerate(sentences, 1):\n"
    '    print(f"[{len(s):>3} chars] sentence {i}: {s}")'
))

cells.append(md(
    "Every segment lands inside the `[80, 160]` character window, yet the cuts still fall on real sentence "
    "boundaries rather than mid-word — that's the boundary *adjustment* at work. Tighten or loosen "
    "`minSentenceLength` / `maxSentenceLength` to trade segment granularity against your downstream constraints."
))

cells.append(md(
    "## Parameter reference\n\n"
    "| Parameter | Default | What it does |\n"
    "| --- | --- | --- |\n"
    "| `threshold` | `0.25` | Boundary probability cutoff (use `0.025` for `sat-12l`). Ignored when a length bound is set. |\n"
    "| `minSentenceLength` | `0` | Minimum characters per sentence. Activates length-constrained (Viterbi) segmentation. |\n"
    "| `maxSentenceLength` | `0` | Maximum characters per sentence. Activates length-constrained (Viterbi) segmentation. |\n"
    "| `blockSize` | `510` | Real sub-word tokens per ONNX window (max 510 for XLM-R). |\n"
    "| `stride` | `256` | Token stride between overlapping windows (smaller = more overlap, smoother boundaries). |\n"
    "| `satBatchSize` | `8` | Windows per ONNX forward pass. |\n"
    "| `weighting` | `\"hat\"` | Overlap weighting: `\"hat\"` (trust window centres) or `\"uniform\"`. |\n"
    "| `trimWhitespace` | `True` | Strip leading/trailing whitespace from each sentence. |\n"
    "| `explodeSentences` | `False` | Put each sentence on its own DataFrame row. |\n\n"
    "That's it! You've exported a SaT model to ONNX, imported it into Spark NLP 🚀, benchmarked it against the "
    "rule-based and DL detectors on long passages in six languages, and used length-constrained segmentation."
))

# Inject EN_TEXT definition for the min/max cell (reuse the English passage).
# We prepend it to that code cell's source instead of relying on a separate var.
for c in cells:
    if c["cell_type"] == "code" and "EN_TEXT" in c["source"]:
        c["source"] = c["source"].replace(
            "sat_bounded = sat_loaded",
            "EN_TEXT = " + json.dumps(EN, ensure_ascii=False) + "\n\nsat_bounded = sat_loaded",
            1,
        )

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

OUT = "examples/python/transformers/onnx/HuggingFace_ONNX_in_Spark_NLP_SentenceDetectorSaT.ipynb"
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("\nwrote", OUT, "with", len(cells), "cells")
