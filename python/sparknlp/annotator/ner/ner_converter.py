#  Copyright 2017-2022 John Snow Labs
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
"""Contains classes for the NerConverter."""

from sparknlp.common import *


class NerConverter(AnnotatorModel):
    """Converts a IOB or IOB2 representation of NER to a user-friendly one, by
    associating the tokens of recognized entities and their label. Results in
    ``CHUNK`` Annotation type.

    NER chunks can then be filtered by setting a whitelist with
    ``setWhiteList``. Chunks with no associated entity (tagged "O") are
    filtered.

    See also `Inside–outside–beginning (tagging)
    <https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)>`__
    for more information.

    ================================= ======================
    Input Annotation types            Output Annotation type
    ================================= ======================
    ``DOCUMENT, TOKEN, NAMED_ENTITY`` ``CHUNK``
    ================================= ======================

    Parameters
    ----------
    whiteList
        If defined, list of entities to process. The rest will be ignored. Do
        not include IOB prefix on labels
    preservePosition
        Whether to preserve the original position of the tokens in the original document
        or use the modified tokens, by default `True`

    Examples
    --------
    This is a continuation of the example of the :class:`.NerDLModel`. See that
    class on how to extract the entities. The output of the NerDLModel follows
    the Annotator schema and can be converted like so:

    >>> result.selectExpr("explode(ner)").show(truncate=False)
    +----------------------------------------------------+
    |col                                                 |
    +----------------------------------------------------+
    |[named_entity, 0, 2, B-ORG, [word -> U.N], []]      |
    |[named_entity, 3, 3, O, [word -> .], []]            |
    |[named_entity, 5, 12, O, [word -> official], []]    |
    |[named_entity, 14, 18, B-PER, [word -> Ekeus], []]  |
    |[named_entity, 20, 24, O, [word -> heads], []]      |
    |[named_entity, 26, 28, O, [word -> for], []]        |
    |[named_entity, 30, 36, B-LOC, [word -> Baghdad], []]|
    |[named_entity, 37, 37, O, [word -> .], []]          |
    +----------------------------------------------------+

    After the converter is used:

    >>> converter = NerConverter() \\
    ...     .setInputCols(["sentence", "token", "ner"]) \\
    ...     .setOutputCol("entities")
    >>> converter.transform(result).selectExpr("explode(entities)").show(truncate=False)
    +------------------------------------------------------------------------+
    |col                                                                     |
    +------------------------------------------------------------------------+
    |[chunk, 0, 2, U.N, [entity -> ORG, sentence -> 0, chunk -> 0], []]      |
    |[chunk, 14, 18, Ekeus, [entity -> PER, sentence -> 0, chunk -> 1], []]  |
    |[chunk, 30, 36, Baghdad, [entity -> LOC, sentence -> 0, chunk -> 2], []]|
    +------------------------------------------------------------------------+
    """
    name = 'NerConverter'

    inputAnnotatorTypes = [AnnotatorType.DOCUMENT, AnnotatorType.TOKEN, AnnotatorType.NAMED_ENTITY]

    outputAnnotatorType = AnnotatorType.CHUNK

    whiteList = Param(
        Params._dummy(),
        "whiteList",
        "If defined, list of entities to process. The rest will be ignored. Do not include IOB prefix on labels",
        typeConverter=TypeConverters.toListString
    )

    preservePosition = Param(
        Params._dummy(),
        "preservePosition",
        "Whether to preserve the original position of the tokens in the original document or use the modified tokens",
        typeConverter=TypeConverters.toBoolean
    )

    nerHasNoSchema = Param(
        Params._dummy(),
        "nerHasNoSchema",
        "set this to true if your NER tags coming from a model that does not have a IOB/IOB2 schema",
        typeConverter=TypeConverters.toBoolean
    )

    def setWhiteList(self, entities):
        """Sets list of entities to process. The rest will be ignored.

        Does not include IOB prefix on labels.

        Parameters
        ----------
        entities : List[str]
            If defined, list of entities to process. The rest will be ignored.

        """
        return self._set(whiteList=entities)

    def setPreservePosition(self, value):
        """
        Whether to preserve the original position of the tokens in the original document
        or use the modified tokens, by default `True`.

        Parameters
        ----------
        value : bool
            Whether to preserve the original position of the tokens in the original
            document or use the modified tokens
        """
        return self._set(preservePosition=value)

    def setNerHasNoSchema(self, value):
        """
        set this to true if your NER tags coming from a model that does not have a IOB/IOB2 schema

        Parameters
        ----------
        value : bool
            set this to true if your NER tags coming from a model that does not have a IOB/IOB2 schema
        """
        return self._set(nerHasNoSchema=value)

    @keyword_only
    def __init__(self):
        super(NerConverter, self).__init__(
            classname="com.johnsnowlabs.nlp.annotators.ner.NerConverter")
