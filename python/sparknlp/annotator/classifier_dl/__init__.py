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

"""Module of annotators for text classification."""
from sparknlp.annotator.classifier_dl.albert_for_sequence_classification import *
from sparknlp.annotator.classifier_dl.albert_for_token_classification import *
from sparknlp.annotator.classifier_dl.albert_for_question_answering import *
from sparknlp.annotator.classifier_dl.bert_for_sequence_classification import *
from sparknlp.annotator.classifier_dl.bert_for_token_classification import *
from sparknlp.annotator.classifier_dl.bert_for_question_answering import *
from sparknlp.annotator.classifier_dl.classifier_dl import *
from sparknlp.annotator.classifier_dl.deberta_for_sequence_classification import *
from sparknlp.annotator.classifier_dl.deberta_for_token_classification import *
from sparknlp.annotator.classifier_dl.deberta_for_question_answering import *
from sparknlp.annotator.classifier_dl.distil_bert_for_sequence_classification import *
from sparknlp.annotator.classifier_dl.distil_bert_for_token_classification import *
from sparknlp.annotator.classifier_dl.distil_bert_for_question_answering import *
from sparknlp.annotator.classifier_dl.longformer_for_sequence_classification import *
from sparknlp.annotator.classifier_dl.longformer_for_token_classification import *
from sparknlp.annotator.classifier_dl.longformer_for_question_answering import *
from sparknlp.annotator.classifier_dl.multi_classifier_dl import *
from sparknlp.annotator.classifier_dl.roberta_for_sequence_classification import *
from sparknlp.annotator.classifier_dl.roberta_for_token_classification import *
from sparknlp.annotator.classifier_dl.roberta_for_question_answering import *
from sparknlp.annotator.classifier_dl.sentiment_dl import *
from sparknlp.annotator.classifier_dl.xlm_roberta_for_sequence_classification import *
from sparknlp.annotator.classifier_dl.xlm_roberta_for_token_classification import *
from sparknlp.annotator.classifier_dl.xlm_roberta_for_question_answering import *
from sparknlp.annotator.classifier_dl.xlnet_for_sequence_classification import *
from sparknlp.annotator.classifier_dl.xlnet_for_token_classification import *
from sparknlp.annotator.classifier_dl.camembert_for_token_classification import *
from sparknlp.annotator.classifier_dl.tapas_for_question_answering import *
from sparknlp.annotator.classifier_dl.camembert_for_sequence_classification import *
from sparknlp.annotator.classifier_dl.camembert_for_question_answering import *
from sparknlp.annotator.classifier_dl.bert_for_zero_shot_classification import *
from sparknlp.annotator.classifier_dl.distil_bert_for_zero_shot_classification import *
from sparknlp.annotator.classifier_dl.roberta_for_zero_shot_classification import *
from sparknlp.annotator.classifier_dl.xlm_roberta_for_zero_shot_classification import *
from sparknlp.annotator.classifier_dl.bart_for_zero_shot_classification import *
from sparknlp.annotator.classifier_dl.deberta_for_zero_shot_classification import *
from sparknlp.annotator.classifier_dl.mpnet_for_sequence_classification import *
from sparknlp.annotator.classifier_dl.mpnet_for_question_answering import *
from sparknlp.annotator.classifier_dl.mpnet_for_token_classification import *
from sparknlp.annotator.classifier_dl.albert_for_zero_shot_classification import *
from sparknlp.annotator.classifier_dl.camembert_for_zero_shot_classification import *
from sparknlp.annotator.classifier_dl.bert_for_multiple_choice import *
from sparknlp.annotator.classifier_dl.xlm_roberta_for_multiple_choice import *
from sparknlp.annotator.classifier_dl.roberta_for_multiple_choice import *
from sparknlp.annotator.classifier_dl.distilbert_for_multiple_choice import *
from sparknlp.annotator.classifier_dl.albert_for_multiple_choice import *
