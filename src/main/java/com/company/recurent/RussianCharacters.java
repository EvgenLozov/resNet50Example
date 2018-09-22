package com.company.recurent;

import java.util.LinkedList;
import java.util.List;

public class RussianCharacters {

    public char[] chars(){
        List<Character> validChars = new LinkedList<>();
        for(char c='а'; c<='я'; c++) validChars.add(c);
        for(char c='А'; c<='Я'; c++) validChars.add(c);
        for(char c='0'; c<='9'; c++) validChars.add(c);
        char[] temp = {'!', '&', '(', ')', '?', '-', '\'', '"', ',', '.', ':', ';', ' ', '\n', '\t'};
        for( char c : temp ) validChars.add(c);
        char[] out = new char[validChars.size()];
        int i=0;
        for( Character c : validChars ) out[i++] = c;
        return out;
    }
}
