package com.johnsnowlabs.nlp.annotators.parser.typdep;

public class Conll09Data {

    private String form;
    private String lemma;
    private String pos;
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

    public String getPos() {
        return pos;
    }

    public void setPos(String pos) {
        this.pos = pos;
    }

    public String getDeprel() {
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

    public Conll09Data(String form, String lemma, String pos, String deprel, int head, int begin, int end) {
        this.form = form;
        this.lemma = lemma;
        this.pos = pos;
        this.deprel = deprel;
        this.head = head;
        this.begin = begin;
        this.end = end;
    }
}
