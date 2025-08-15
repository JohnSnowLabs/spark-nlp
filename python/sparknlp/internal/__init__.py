#  Copyright 2017-2023 John Snow Labs
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
"""Module of Spark NLP internal classes for annotator development."""
from pyspark.ml import PipelineModel

from sparknlp.internal.annotator_java_ml import *
from sparknlp.internal.annotator_transformer import *
from sparknlp.internal.extended_java_wrapper import *
from sparknlp.internal.params_getters_setters import *
from sparknlp.internal.recursive import *


# Wrapper Definitions
class _AlbertLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AlbertLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.AlbertEmbeddings.loadSavedModel",
            path,
            jspark,
        )


class _AlbertSequenceClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AlbertSequenceClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.AlbertForSequenceClassification.loadSavedModel",
            path,
            jspark,
        )


class _AlbertTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AlbertTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.AlbertForTokenClassification.loadSavedModel",
            path,
            jspark,
        )


class _AlbertQuestionAnsweringLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AlbertQuestionAnsweringLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.AlbertForQuestionAnswering.loadSavedModel",
            path,
            jspark,
        )


class _AlbertForZeroShotClassificationLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AlbertForZeroShotClassificationLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.AlbertForZeroShotClassification.loadSavedModel",
            path,
            jspark,
        )


class _AlbertMultipleChoiceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AlbertMultipleChoiceLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.AlbertForMultipleChoice.loadSavedModel",
            path,
            jspark,
        )


class _BertLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_BertLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.BertEmbeddings.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )


class _BertSentenceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_BertSentenceLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.BertSentenceEmbeddings.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )


class _BertSequenceClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BertSequenceClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.BertForSequenceClassification.loadSavedModel",
            path,
            jspark,
        )


class _BertTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BertTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.BertForTokenClassification.loadSavedModel",
            path,
            jspark,
        )


class _BertQuestionAnsweringLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BertQuestionAnsweringLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.BertForQuestionAnswering.loadSavedModel",
            path,
            jspark,
        )

class _BertMultipleChoiceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BertMultipleChoiceLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.BertForMultipleChoice.loadSavedModel",
            path,
            jspark,
        )

class _CoHereLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_CoHereLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.CoHereTransformer.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )

class _DeBERTaLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DeBERTaLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.DeBertaEmbeddings.loadSavedModel",
            path,
            jspark,
        )


class _DeBertaSequenceClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DeBertaSequenceClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.DeBertaForSequenceClassification.loadSavedModel",
            path,
            jspark,
        )


class _DeBertTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DeBertTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.DeBertaForTokenClassification.loadSavedModel",
            path,
            jspark,
        )


class _DeBertaQuestionAnsweringLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DeBertaQuestionAnsweringLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.DeBertaForQuestionAnswering.loadSavedModel",
            path,
            jspark,
        )


class _CamemBertLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_CamemBertLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.CamemBertEmbeddings.loadSavedModel",
            path,
            jspark,
        )

class _CPMLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_CPMLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.CPMTransformer.loadSavedModel",
            path,
            jspark,
            use_openvino
        )


class _DistilBertLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DistilBertLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.DistilBertEmbeddings.loadSavedModel",
            path,
            jspark,
        )


class _DistilBertSequenceClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DistilBertSequenceClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForSequenceClassification.loadSavedModel",
            path,
            jspark,
        )


class _DistilBertTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DistilBertTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForTokenClassification.loadSavedModel",
            path,
            jspark,
        )


class _DistilBertQuestionAnsweringLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DistilBertQuestionAnsweringLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForQuestionAnswering.loadSavedModel",
            path,
            jspark,
        )


class _DistilBertMultipleChoiceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DistilBertMultipleChoiceLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForMultipleChoice.loadSavedModel",
            path,
            jspark,
        )


class _ElmoLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_ElmoLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.ElmoEmbeddings.loadSavedModel",
            path,
            jspark,
        )


class _E5Loader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_E5Loader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.E5Embeddings.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )


class _MiniLMLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_MiniLMLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.MiniLMEmbeddings.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )


class _BGELoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BGELoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.BGEEmbeddings.loadSavedModel", path, jspark
        )


class _GPT2Loader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_GPT2Loader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.GPT2Transformer.loadSavedModel",
            path,
            jspark,
        )

class _Gemma3ForMultiModalLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_Gemma3ForMultiModalLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.Gemma3ForMultiModal.loadSavedModel",
            path,
            jspark,
            use_openvino
        )

class _InternVLForMultiModalLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_InternVLForMultiModalLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.InternVLForMultiModal.loadSavedModel",
            path,
            jspark,
            use_openvino
        )


class _JanusForMultiModalLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_JanusForMultiModalLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.JanusForMultiModal.loadSavedModel",
            path,
            jspark,
            use_openvino
        )

class _LLAMA2Loader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_LLAMA2Loader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.LLAMA2Transformer.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )

class _LLAMA3Loader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_LLAMA3Loader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.LLAMA3Transformer.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )

class _LongformerLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_LongformerLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.LongformerEmbeddings.loadSavedModel",
            path,
            jspark,
        )


class _LongformerSequenceClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_LongformerSequenceClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.LongformerForSequenceClassification.loadSavedModel",
            path,
            jspark,
        )


class _LongformerTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_LongformerTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.LongformerForTokenClassification.loadSavedModel",
            path,
            jspark,
        )


class _LongformerQuestionAnsweringLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_LongformerQuestionAnsweringLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.LongformerForQuestionAnswering.loadSavedModel",
            path,
            jspark,
        )

class _LLAVAForMultiModalLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_LLAVAForMultiModalLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.LLAVAForMultiModal.loadSavedModel",
            path,
            jspark,
            use_openvino
        )

class _M2M100Loader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_M2M100Loader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.M2M100Transformer.loadSavedModel",
            path,
            jspark,
        )


class _MistralLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_MistralLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.MistralTransformer.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )

class _MLLamaForMultimodalLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_MLLamaForMultimodalLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.MLLamaForMultimodal.loadSavedModel",
            path,
            jspark,
            use_openvino
        )

class _NLLBLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark,  use_openvino=False):
        super(_NLLBLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.NLLBTransformer.loadSavedModel",
            path,
            jspark,
            use_openvino)

class _MarianLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_MarianLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.MarianTransformer.loadSavedModel",
            path,
            jspark,
        )


class _MPNetLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_MPNetLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.MPNetEmbeddings.loadSavedModel",
            path,
            jspark,
        )


class _OLMoLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_OLMoLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.OLMoTransformer.loadSavedModel", path, jspark)
class _Phi2Loader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_Phi2Loader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.Phi2Transformer.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )

class _Phi3Loader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_Phi3Loader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.Phi3Transformer.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )

class _Phi3VisionLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_Phi3VisionLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.Phi3Vision.loadSavedModel",
            path,
            jspark,
            use_openvino
        )

class _RoBertaLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_RoBertaLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.RoBertaEmbeddings.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )


class _RoBertaSentenceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_RoBertaSentenceLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.RoBertaSentenceEmbeddings.loadSavedModel",
            path,
            jspark,
        )


class _RoBertaSequenceClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_RoBertaSequenceClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForSequenceClassification.loadSavedModel",
            path,
            jspark,
        )


class _RoBertaTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_RoBertaTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForTokenClassification.loadSavedModel",
            path,
            jspark,
        )


class _RoBertaQuestionAnsweringLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_RoBertaQuestionAnsweringLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForQuestionAnswering.loadSavedModel",
            path,
            jspark,
        )


class _RoBertaMultipleChoiceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_RoBertaMultipleChoiceLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForMultipleChoice.loadSavedModel",
            path,
            jspark,
        )


class _StarCoderLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_StarCoderLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.StarCoderTransformer.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )

class _T5Loader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_T5Loader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.T5Transformer.loadSavedModel",
            path,
            jspark,
        )


class _BartLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, useCache):
        super(_BartLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.BartTransformer.loadSavedModel",
            path,
            jspark,
            useCache,
        )


class _NomicLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_NomicLoader, self).__init__("com.johnsnowlabs.nlp.embeddings.NomicEmbeddings.loadSavedModel", path, jspark, use_openvino)


class _QwenLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_QwenLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.QwenTransformer.loadSavedModel", path, jspark, use_openvino)


class _USELoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, loadsp):
        super(_USELoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.UniversalSentenceEncoder.loadSavedModel",
            path,
            jspark,
            loadsp,
        )


class _XlmRoBertaLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_XlmRoBertaLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.XlmRoBertaEmbeddings.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )


class _XlmRoBertaSentenceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlmRoBertaSentenceLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.XlmRoBertaSentenceEmbeddings.loadSavedModel",
            path,
            jspark,
        )


class _XlmRoBertaSequenceClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlmRoBertaSequenceClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForSequenceClassification.loadSavedModel",
            path,
            jspark,
        )


class _XlmRoBertaTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlmRoBertaTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForTokenClassification.loadSavedModel",
            path,
            jspark,
        )


class _XlmRoBertaQuestionAnsweringLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlmRoBertaQuestionAnsweringLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForQuestionAnswering.loadSavedModel",
            path,
            jspark,
        )


class _XlmRoBertaMultipleChoiceLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlmRoBertaMultipleChoiceLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForMultipleChoice.loadSavedModel",
            path,
            jspark,
        )


class _XlnetLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlnetLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.XlnetEmbeddings.loadSavedModel",
            path,
            jspark,
        )


class _XlnetSequenceClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlnetSequenceClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.XlnetForSequenceClassification.loadSavedModel",
            path,
            jspark,
        )


class _XlnetTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlnetTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.XlnetForTokenClassification.loadSavedModel",
            path,
            jspark,
        )


class _ClearCache(ExtendedJavaWrapper):
    def __init__(self, name, language, remote_loc):
        super(_ClearCache, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.clearCache",
            name,
            language,
            remote_loc,
        )


class _CoNLLGeneratorExportFromTargetAndPipeline(ExtendedJavaWrapper):
    def __init__(self, spark, target, pipeline, output_path):
        if type(pipeline) == PipelineModel:
            pipeline = pipeline._to_java()
        elif type(pipeline) == str:
            pipeline = PipelineModel.load(pipeline)._to_java()
        if type(target) == DataFrame:
            super(_CoNLLGeneratorExportFromTargetAndPipeline, self).__init__(
                "com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles",
                target._jdf,
                pipeline,
                output_path,
            )
        else:
            super(_CoNLLGeneratorExportFromTargetAndPipeline, self).__init__(
                "com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles",
                spark._jsparkSession,
                target,
                pipeline,
                output_path,
            )


class _CoNLLGeneratorExportFromDataFrameAndField(ExtendedJavaWrapper):

    def __init__(self, dataframe, output_path, metadata_sentence_key):
        super(_CoNLLGeneratorExportFromDataFrameAndField, self).__init__(
            "com.johnsnowlabs.util.CoNLLGenerator.exportConllFilesFromField",
            dataframe,
            output_path,
            metadata_sentence_key,
        )


class _CoNLLGeneratorExportFromDataFrame(ExtendedJavaWrapper):
    def __init__(self, dataframe, output_path):
        super(_CoNLLGeneratorExportFromDataFrame, self).__init__(
            "com.johnsnowlabs.util.CoNLLGenerator.exportConllFiles",
            dataframe,
            output_path,
        )


class _CoverageResult(ExtendedJavaWrapper):
    def __init__(self, covered, total, percentage):
        super(_CoverageResult, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.CoverageResult", covered, total, percentage
        )


class _DownloadModelDirectly(ExtendedJavaWrapper):
    def __init__(self, name, remote_loc="public/models", unzip=True):
        super(_DownloadModelDirectly, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.downloadModelDirectly",
            name,
            remote_loc,
            unzip,
        )


class _DownloadModel(ExtendedJavaWrapper):
    def __init__(self, reader, name, language, remote_loc, validator):
        super(_DownloadModel, self).__init__(
            "com.johnsnowlabs.nlp.pretrained." + validator + ".downloadModel",
            reader,
            name,
            language,
            remote_loc,
        )


class _DownloadPipeline(ExtendedJavaWrapper):
    def __init__(self, name, language, remote_loc):
        super(_DownloadPipeline, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.downloadPipeline",
            name,
            language,
            remote_loc,
        )


class _DownloadPredefinedPipeline(ExtendedJavaWrapper):
    def __init__(self, java_path):
        super(_DownloadPredefinedPipeline, self).__init__(java_path)


class _EmbeddingsCoverageColumn(ExtendedJavaWrapper):
    def __init__(self, dataset, embeddings_col, output_col):
        super(_EmbeddingsCoverageColumn, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel.withCoverageColumn",
            dataset._jdf,
            embeddings_col,
            output_col,
        )


class _EmbeddingsOverallCoverage(ExtendedJavaWrapper):
    def __init__(self, dataset, embeddings_col):
        super(_EmbeddingsOverallCoverage, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.WordEmbeddingsModel.overallCoverage",
            dataset._jdf,
            embeddings_col,
        )


class _ExternalResource(ExtendedJavaWrapper):
    def __init__(self, path, read_as, options):
        super(_ExternalResource, self).__init__(
            "com.johnsnowlabs.nlp.util.io.ExternalResource.fromJava",
            path,
            read_as,
            options,
        )


class _ConfigLoaderGetter(ExtendedJavaWrapper):
    def __init__(self):
        super(_ConfigLoaderGetter, self).__init__(
            "com.johnsnowlabs.util.ConfigLoader.getConfigPath"
        )


class _GetResourceSize(ExtendedJavaWrapper):
    def __init__(self, name, language, remote_loc):
        super(_GetResourceSize, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.getDownloadSize",
            name,
            language,
            remote_loc,
        )


class _LightPipeline(ExtendedJavaWrapper):
    def __init__(self, pipelineModel, parse_embeddings):
        super(_LightPipeline, self).__init__(
            "com.johnsnowlabs.nlp.LightPipeline",
            pipelineModel._to_java(),
            parse_embeddings,
        )


class _RegexRule(ExtendedJavaWrapper):
    def __init__(self, rule, identifier):
        super(_RegexRule, self).__init__(
            "com.johnsnowlabs.nlp.util.regex.RegexRule", rule, identifier
        )


class _ShowAvailableAnnotators(ExtendedJavaWrapper):
    def __init__(self):
        super(_ShowAvailableAnnotators, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.showAvailableAnnotators"
        )


class _ShowPublicModels(ExtendedJavaWrapper):
    def __init__(self, annotator, lang, version):
        super(_ShowPublicModels, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.showPublicModels",
            annotator,
            lang,
            version,
        )


class _ShowPublicPipelines(ExtendedJavaWrapper):
    def __init__(self, lang, version):
        super(_ShowPublicPipelines, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.showPublicPipelines",
            lang,
            version,
        )


class _ShowUnCategorizedResources(ExtendedJavaWrapper):
    def __init__(self):
        super(_ShowUnCategorizedResources, self).__init__(
            "com.johnsnowlabs.nlp.pretrained.PythonResourceDownloader.showUnCategorizedResources"
        )


class _StorageHelper(ExtendedJavaWrapper):
    def __init__(self, path, spark, database, storage_ref, within_storage):
        super(_StorageHelper, self).__init__(
            "com.johnsnowlabs.storage.StorageHelper.load",
            path,
            spark._jsparkSession,
            database,
            storage_ref,
            within_storage,
        )


class _SpanBertCorefLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_SpanBertCorefLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.coref.SpanBertCorefModel.loadSavedModel",
            path,
            jspark,
        )


class _NerDLGraphBuilder(ExtendedJavaWrapper):
    def __init__(self, dataset, input_col, label_col):
        super(_NerDLGraphBuilder, self).__init__(
            "com.johnsnowlabs.nlp.annotators.ner.dl.NerDLApproach.getGraphParams",
            dataset,
            input_col,
            label_col,
        )


class _ResourceHelper_moveFile(ExtendedJavaWrapper):
    def __init__(self, local_file, hdfs_file):
        super(_ResourceHelper_moveFile, self).__init__(
            "com.johnsnowlabs.nlp.util.io.ResourceHelper.moveFile",
            local_file,
            hdfs_file,
        )


class _ResourceHelper_validFile(ExtendedJavaWrapper):
    def __init__(self, path):
        super(_ResourceHelper_validFile, self).__init__(
            "com.johnsnowlabs.nlp.util.io.ResourceHelper.validFile", path
        )


class _ViTForImageClassification(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_ViTForImageClassification, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.ViTForImageClassification.loadSavedModel",
            path,
            jspark,
        )


class _VisionEncoderDecoderForImageCaptioning(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_VisionEncoderDecoderForImageCaptioning, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.VisionEncoderDecoderForImageCaptioning.loadSavedModel",
            path,
            jspark,
        )


class _SwinForImageClassification(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_SwinForImageClassification, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.SwinForImageClassification.loadSavedModel",
            path,
            jspark,
        )


class _ConvNextForImageClassification(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_ConvNextForImageClassification, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.ConvNextForImageClassification.loadSavedModel",
            path,
            jspark,
        )


class _Wav2Vec2ForCTC(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_Wav2Vec2ForCTC, self).__init__(
            "com.johnsnowlabs.nlp.annotators.audio.Wav2Vec2ForCTC.loadSavedModel",
            path,
            jspark,
        )


class _HubertForCTC(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_HubertForCTC, self).__init__(
            "com.johnsnowlabs.nlp.annotators.audio.HubertForCTC.loadSavedModel",
            path,
            jspark,
        )


class _WhisperForCTC(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_WhisperForCTC, self).__init__(
            "com.johnsnowlabs.nlp.annotators.audio.WhisperForCTC.loadSavedModel",
            path,
            jspark,
        )


class _CamemBertForTokenClassificationLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_CamemBertForTokenClassificationLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.CamemBertForTokenClassification.loadSavedModel",
            path,
            jspark,
        )


class _TapasForQuestionAnsweringLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_TapasForQuestionAnsweringLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.TapasForQuestionAnswering.loadSavedModel",
            path,
            jspark,
        )


class _CamemBertForSequenceClassificationLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_CamemBertForSequenceClassificationLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.CamemBertForSequenceClassification.loadSavedModel",
            path,
            jspark,
        )


class _CamemBertQuestionAnsweringLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_CamemBertQuestionAnsweringLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.CamemBertForQuestionAnswering.loadSavedModel",
            path,
            jspark,
        )

class _CamemBertForZeroShotClassificationLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_CamemBertForZeroShotClassificationLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.CamemBertForZeroShotClassification.loadSavedModel",
            path,
            jspark,
        )

class _RobertaQAToZeroShotNerLoader(ExtendedJavaWrapper):
    def __init__(self, path):
        super(_RobertaQAToZeroShotNerLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.ner.dl.ZeroShotNerModel.load", path
        )


class _BertZeroShotClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BertZeroShotClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.BertForZeroShotClassification.loadSavedModel",
            path,
            jspark,
        )


class _DistilBertForZeroShotClassification(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DistilBertForZeroShotClassification, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.DistilBertForZeroShotClassification.loadSavedModel",
            path,
            jspark,
        )


class _RoBertaForZeroShotClassification(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_RoBertaForZeroShotClassification, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.RoBertaForZeroShotClassification.loadSavedModel",
            path,
            jspark,
        )


class _XlmRoBertaForZeroShotClassification(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_XlmRoBertaForZeroShotClassification, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.XlmRoBertaForZeroShotClassification.loadSavedModel",
            path,
            jspark,
        )


class _InstructorLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_InstructorLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.InstructorEmbeddings.loadSavedModel",
            path,
            jspark,
        )


class _BartForZeroShotClassification(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BartForZeroShotClassification, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.BartForZeroShotClassification.loadSavedModel",
            path,
            jspark,
        )


class _CLIPForZeroShotClassification(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_CLIPForZeroShotClassification, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.CLIPForZeroShotClassification.loadSavedModel",
            path,
            jspark,
        )


class _DeBertaForZeroShotClassification(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_DeBertaForZeroShotClassification, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.DeBertaForZeroShotClassification.loadSavedModel",
            path,
            jspark,
        )


class _MPNetForSequenceClassificationLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_MPNetForSequenceClassificationLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.MPNetForSequenceClassification.loadSavedModel",
            path,
            jspark,
        )


class _MPNetForQuestionAnsweringLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_MPNetForQuestionAnsweringLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.MPNetForQuestionAnswering.loadSavedModel",
            path,
            jspark,
        )


class _MPNetForTokenClassifierLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_MPNetForTokenClassifierLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.classifier.dl.MPNetForTokenClassification.loadSavedModel",
            path,
            jspark,
        )


class _UAEEmbeddingsLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_UAEEmbeddingsLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.UAEEmbeddings.loadSavedModel", path, jspark
        )


class _AutoGGUFLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AutoGGUFLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFModel.loadSavedModel", path, jspark)


class _MxbaiEmbeddingsLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_MxbaiEmbeddingsLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.MxbaiEmbeddings.loadSavedModel", path, jspark
        )


class _SnowFlakeEmbeddingsLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_SnowFlakeEmbeddingsLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.SnowFlakeEmbeddings.loadSavedModel", path, jspark
        )


class _AutoGGUFEmbeddingsLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AutoGGUFEmbeddingsLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.AutoGGUFEmbeddings.loadSavedModel", path, jspark)


class _BLIPForQuestionAnswering(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_BLIPForQuestionAnswering, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.BLIPForQuestionAnswering.loadSavedModel",
            path,
            jspark,
        )


class _AutoGGUFVisionLoader(ExtendedJavaWrapper):
    def __init__(self, modelPath, mmprojPath, jspark):
        super(_AutoGGUFVisionLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFVisionModel.loadSavedModel", modelPath, mmprojPath, jspark)
        
               
class _Qwen2VLTransformerLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_Qwen2VLTransformerLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.Qwen2VLTransformer.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )

class _PaliGemmaForMultiModalLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_PaliGemmaForMultiModalLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.PaliGemmaForMultiModal.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )

class _SmolVLMTransformerLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_SmolVLMTransformerLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.SmolVLMTransformer.loadSavedModel",
            path,
            jspark,
            use_openvino
        )

class _Florence2TransformerLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_Florence2TransformerLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.cv.Florence2Transformer.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )
class _E5VEmbeddingsLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_E5VEmbeddingsLoader, self).__init__(
            "com.johnsnowlabs.nlp.embeddings.E5VEmbeddings.loadSavedModel",
            path,
            jspark,
            use_openvino
        )

class _Phi4Loader(ExtendedJavaWrapper):
    def __init__(self, path, jspark, use_openvino=False):
        super(_Phi4Loader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.Phi4Transformer.loadSavedModel",
            path,
            jspark,
            use_openvino,
        )

class _AutoGGUFRerankerLoader(ExtendedJavaWrapper):
    def __init__(self, path, jspark):
        super(_AutoGGUFRerankerLoader, self).__init__(
            "com.johnsnowlabs.nlp.annotators.seq2seq.AutoGGUFReranker.loadSavedModel", path, jspark)