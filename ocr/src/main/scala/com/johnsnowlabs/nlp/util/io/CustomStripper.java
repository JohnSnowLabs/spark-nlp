package com.johnsnowlabs.nlp.util.io;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.pdfbox.text.TextPosition;

/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/**
 *
 * @author samue
 */
public class CustomStripper extends PDFTextStripper {

    public CustomStripper() throws IOException {

    }

    @Override
    protected void startPage(PDPage page) throws IOException
    {
        startOfLine = true;
        super.startPage(page);
    }

    @Override
    protected void writeLineSeparator() throws IOException
    {
        startOfLine = true;
        super.writeLineSeparator();
    }

    @Override
    public String getText(PDDocument doc) throws IOException
    {
        lines = new ArrayList<TextLine>();
        return super.getText(doc);
    }

    @Override
    protected void writeWordSeparator() throws IOException
    {
        TextLine tmpline = null;

        tmpline = lines.get(lines.size() - 1);
        tmpline.text += getWordSeparator();

        super.writeWordSeparator();
    }


    @Override
    protected void writeString(String text, List<TextPosition> textPositions) throws IOException
    {
        TextLine tmpline = null;

        if (startOfLine) {
            tmpline = new TextLine();
            tmpline.text = text;
            tmpline.textPositions = textPositions;
            lines.add(tmpline);
        } else {
            tmpline = lines.get(lines.size() - 1);
            tmpline.text += text;
            tmpline.textPositions.addAll(textPositions);
        }

        if (startOfLine)
        {
            startOfLine = false;
        }
        super.writeString(text, textPositions);
    }

    boolean startOfLine = true;
    public ArrayList<TextLine> lines = null;

}



