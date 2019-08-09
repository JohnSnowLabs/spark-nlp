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
