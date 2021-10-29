/*
 * Copyright 2017-2021 John Snow Labs
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

package com.johnsnowlabs.nlp.annotators.parser.typdep;

public class ConllData {

    private String form;
    private String lemma;
    private String uPos;

    public String getXPos() {
        return xPos;
    }

    public void setXPos(String xPos) {
        this.xPos = xPos;
    }

    public String getUPos() {
        return uPos;
    }

    public void setUPos(String uPos) {
        this.uPos = uPos;
    }

    private String xPos;
    private String deprel;
    private int head;
    private int begin;
    private int end;

    public String getForm() {
        return form;
    }

    public void setForm(String form) {
        this.form = form;
    }

    public String getLemma() {
        return lemma;
    }

    public void setLemma(String lemma) {
        this.lemma = lemma;
    }

    public String getDepRel() {
        return deprel;
    }

    public int getHead() {
        return head;
    }

    public void setHead(int head) {
        this.head = head;
    }

    public int getBegin() {
        return begin;
    }

    public void setBegin(int begin) {
        this.begin = begin;
    }

    public int getEnd() {
        return end;
    }

    public void setEnd(int end) {
        this.end = end;
    }

    public ConllData(String form, String lemma, String uPos, String xPos, String deprel, int head, int begin, int end) {
        this.form = form;
        this.lemma = lemma;
        this.uPos = uPos;
        this.xPos = xPos;
        this.deprel = deprel;
        this.head = head;
        this.begin = begin;
        this.end = end;
    }
}
