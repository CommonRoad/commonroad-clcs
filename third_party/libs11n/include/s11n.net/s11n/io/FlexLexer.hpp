#ifndef __FLEX_LEXER_HPP
#define __FLEX_LEXER_HPP
// FlexLexer.h[pp] -- define interfaces for lexical analyzer classes generated
//                by flex

// Copyright (c) 1993 The Regents of the University of California.
// All rights reserved.
//
// This code is derived from software contributed to Berkeley by
// Kent Williams and Tom Epperly.
//
// Redistribution and use in source and binary forms with or without
// modification are permitted provided that: (1) source distributions retain
// this entire copyright notice and comment, and (2) distributions including
// binaries display the following acknowledgement:  ``This product includes
// software developed by the University of California, Berkeley and its
// contributors'' in the documentation or other materials provided with the
// distribution and in all advertising materials mentioning features or use
// of this software.  Neither the name of the University nor the names of
// its contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.

// This file defines FlexLexer, an abstract class which specifies the
// external interface provided to flex C++ lexer objects, and yyFlexLexer,
// which defines a particular lexer class.
//
// If you want to create multiple lexer classes, you use the -P flag
// to rename each yyFlexLexer to some other xxFlexLexer.  You then
// include <FlexLexer.h> in your other sources once per lexer class:
//
//      #undef yyFlexLexer
//      #define yyFlexLexer xxFlexLexer
//      #include <FlexLexer.h>
//
//      #undef yyFlexLexer
//      #define yyFlexLexer zzFlexLexer
//      #include <FlexLexer.h>
//      ...


#include <iostream>

//////////////////////////////////////////////////////////////////////
// workarounds for the (very outdated) flex output:
using std::istream;
using std::ostream;
using std::cin;
using std::cout;
using std::cerr;
using std::endl;
//////////////////////////////////////////////////////////////////////

extern "C++" {

struct yy_buffer_state;
typedef int yy_state_type;

/**
From:
Frank P.E. Vanris
Subject: Modification of FlexLexer.h

Date: Thu, 17 Jan 2002 14:35:26 -0600

I modified FlexLexer.h a bit.

To prevent a compiler warning (complaining that yylex() was hiding this
function) I added in class yyFlexLexer the following:


        int yylex( istream* new_in, ostream* new_out = 0 ) {
          return FlexLexer::yylex(new_in, new_out);
        }

Also in the class yyFlexLexer I added a protected method:

        // Function that can be used by subclasses during yylex()
        virtual int actionHook(void* data = 0) {
          return data == 0;
        }

I override this function in a subclass of yyFlexLexer and I use it in
the lex file to call my subclass. Any data I have to save (e.g.
character count) I can now keep in my own subclass as memberfields
instead of global variables in the lex file.

I attached the FlexLexer.h file.

I don't know whether Flex is still maintained since the last tarfile is
from July 27th 1997, but I thought I at least pass it on.

Frank Vanris.



Sept 2003: changes by stephan@s11n.net
   - added 'using' statements for istream/ostreams, so gcc 3.x can play, too.
   - replace iostream.h with iostream, so gcc 3.x won't bitch about backwards-compatibility headers.

Oct 2004: changed by stephan@s11n.net
   - Removed the macro-based generation of classes (was too
   problematic for multi-lexer projects) and replaced it with a
   template file, which gets filtered at build-time to create the
   subclass.

*/
class FlexLexer {
public:
        virtual ~FlexLexer()    { }

        const char* YYText()    { return yytext; }
        int YYLeng()            { return yyleng; }

        virtual void
                yy_switch_to_buffer( struct yy_buffer_state* new_buffer ) = 0;
        virtual struct yy_buffer_state*
                yy_create_buffer( istream* s, int size ) = 0;
        virtual void yy_delete_buffer( struct yy_buffer_state* b ) = 0;
        virtual void yyrestart( istream* s ) = 0;

        virtual int yylex() = 0;

        // Call yylex with new input/output sources.
        int yylex( istream* new_in, ostream* new_out = 0 )
                {
                switch_streams( new_in, new_out );
                return yylex();
                }

        // Switch to new input/output streams.  A nil stream pointer
        // indicates "keep the current one".
        virtual void switch_streams( istream* new_in = 0,
                                        ostream* new_out = 0 ) = 0;

        int lineno() const              { return yylineno; }

        int debug() const               { return yy_flex_debug; }
        void set_debug( int flag )      { yy_flex_debug = flag; }

protected:
        char* yytext;
        int yyleng;
        int yylineno;           // only maintained if you use %option yylineno
        int yy_flex_debug;      // only has effect with -d or "%option debug"
};

} // extern "C++"

#endif
// ^^^^ multi-include guard

