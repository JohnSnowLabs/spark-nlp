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
from pyspark.sql import SparkSession

from sparknlp.internal import ExtendedJavaWrapper


class SpacyToAnnotation(ExtendedJavaWrapper):

    """Helper class to load a list of tokens/sentences as JSON to Annotation.

    The JSON will be in this format:
        [
         {
            "tokens": ["Hello", "world", "!", "How", "are", "you", "today", "?", "I", "'m", "fine", "thanks", "."],
            "token_spaces": [true, false, true, true, true, true, false, true, false, true, true, false, false],
            "sentence_ends": [2, 7, 12]
         }
        ]

    Examples
    --------
    >>> from sparknlp.training import SpacyToAnnotation
    >>> result = SpacyToAnnotation().readDataset(spark, "src/test/resources/spacy-to-annotation/multi_doc_tokens.json")
    >>> result.show(False)
    +-------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |document                                                                             |sentence                                                                                                                                                                      |token                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
    +-------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    |[{document, 0, 55, John went to the store last night. He bought some bread., {}, []}]|[{document, 0, 33, John went to the store last night., {sentence -> 0}, []}, {document, 35, 55, He bought some bread., {sentence -> 1}, []}]                                  |[{token, 0, 3, John, {sentence -> 0}, []}, {token, 5, 8, went, {sentence -> 0}, []}, {token, 10, 11, to, {sentence -> 0}, []}, {token, 13, 15, the, {sentence -> 0}, []}, {token, 17, 21, store, {sentence -> 0}, []}, {token, 23, 26, last, {sentence -> 0}, []}, {token, 28, 32, night, {sentence -> 0}, []}, {token, 33, 33, ., {sentence -> 0}, []}, {token, 35, 36, He, {sentence -> 1}, []}, {token, 38, 43, bought, {sentence -> 1}, []}, {token, 45, 48, some, {sentence -> 1}, []}, {token, 50, 54, bread, {sentence -> 1}, []}, {token, 55, 55, ., {sentence -> 1}, []}]|
    |[{document, 0, 47, Hello world! How are you today? I'm fine thanks., {}, []}]        |[{document, 0, 11, Hello world!, {sentence -> 0}, []}, {document, 13, 30, How are you today?, {sentence -> 1}, []}, {document, 32, 47, I'm fine thanks., {sentence -> 2}, []}]|[{token, 0, 4, Hello, {sentence -> 0}, []}, {token, 6, 10, world, {sentence -> 0}, []}, {token, 11, 11, !, {sentence -> 0}, []}, {token, 13, 15, How, {sentence -> 1}, []}, {token, 17, 19, are, {sentence -> 1}, []}, {token, 21, 23, you, {sentence -> 1}, []}, {token, 25, 29, today, {sentence -> 1}, []}, {token, 30, 30, ?, {sentence -> 1}, []}, {token, 32, 32, I, {sentence -> 2}, []}, {token, 33, 34, 'm, {sentence -> 2}, []}, {token, 36, 39, fine, {sentence -> 2}, []}, {token, 41, 46, thanks, {sentence -> 2}, []}, {token, 47, 47, ., {sentence -> 2}, []}]     |
    +-------------------------------------------------------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

    """

    def __init__(self):
        super(SpacyToAnnotation, self).__init__("com.johnsnowlabs.nlp.training.SpacyToAnnotation")

    def readJsonFile(self, spark, jsonFilePath, params=None):
        if params is None:
            params = {}

        jSession = spark._jsparkSession

        jdf = self._java_obj.readJsonFileJava(jSession, jsonFilePath, params)
        annotation_dataset = self.getDataFrame(spark, jdf)
        return annotation_dataset
