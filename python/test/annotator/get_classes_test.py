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
import unittest

import pytest

from sparknlp.annotator import *


@pytest.mark.slow
class GetClassesTestSpec(unittest.TestCase):

    def runTest(self):
        print(AlbertForTokenClassification.pretrained().getClasses())
        print(XlnetForTokenClassification.pretrained().getClasses())
        print(BertForTokenClassification.pretrained().getClasses())
        print(DistilBertForTokenClassification.pretrained().getClasses())
        print(RoBertaForTokenClassification.pretrained().getClasses())
        print(XlmRoBertaForTokenClassification.pretrained().getClasses())
        print(LongformerForTokenClassification.pretrained().getClasses())

        print(AlbertForSequenceClassification.pretrained().getClasses())
        print(XlnetForSequenceClassification.pretrained().getClasses())
        print(BertForSequenceClassification.pretrained().getClasses())
        print(DistilBertForSequenceClassification.pretrained().getClasses())
        print(RoBertaForSequenceClassification.pretrained().getClasses())
        print(XlmRoBertaForSequenceClassification.pretrained().getClasses())
        print(LongformerForSequenceClassification.pretrained().getClasses())

