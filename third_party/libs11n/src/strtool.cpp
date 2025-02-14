#include <s11n.net/s11n/io/strtool.hpp>
#include <cctype>
#include <algorithm>

namespace s11n { namespace io { namespace strtool {

        std::size_t translate_entities( std::string & str,  const entity_map & map, bool reverse )
        {
                if( str.empty() || ( 0 == map.size() ) ) return 0;
                std::size_t count = 0;
                std::string::size_type pos = str.npos;
                std::string::size_type pos2 = str.npos;
                entity_map::const_iterator mit;
                entity_map::const_iterator met = map.end();
                std::string key;
                std::string val;
                if( reverse )
                {
			// CERR << "Reverse-translating entities.\n";
			// treat KEY=VAL as VAL=KEY
                        mit = map.begin();
                        for( ; mit != met; ++mit )
                        {
                                key = (*mit).second;
                                val = (*mit).first;
				//CERR << "reverse-map key=["<<key<<"] val=["<<val<<"]\n";
                                while( str.npos != (pos = str.rfind( key, pos2 )) )
                                {
                                        ++count;
					pos2 = pos-1;
                                        str.replace( pos, key.size(), val );
                                }
				pos2 = str.size() - 1;
                        }
                }
                else
                {
			//CERR << "Translating entities in ["<<str<<"]...\n";
			// treat KEY=VAL as KEY=VAL
// broken code: causes an endless loop in some cases
//                         mit = map.begin();
// 			pos = 0;
// 			pos2 = str.size()-1;
//                         for( ; mit != met; ++mit )
//                         {
//                                 key = (*mit).first;
//                                 val = (*mit).second;
//  				//CERR << "forward-map key=["<<key<<"] val=["<<val<<"]\n";
//                                 while( str.npos != (pos = str.rfind( key, pos2 )) )
//                                 {
// 					// CERR << "forward-map key=["<<key<<"] val=["<<val<<"]\n";
//                                         ++count;
// 					pos2 = pos-1;
//                                         str.replace( pos, key.size(), val );
//                                 }
// 				pos2 = str.size()-1;
//                         }
// this code works:
                         pos = str.size() - 1;
                         for( ; pos != std::string::npos; --pos )
                         {
                                 mit = map.find( str.substr(pos,1) );
                                 if( met == mit ) continue;
                                 ++count;
                                 str.replace( pos, 1, (*mit).second );
                         }
                }
                return count;
        }

        std::size_t escape_string( std::string & ins, const std::string & to_esc, const std::string & esc )
        {
                std::string::size_type pos;
                pos = ins.find_first_of( to_esc );
                std::size_t reps = 0;
                while( pos != std::string::npos )
                {
                        ins.insert( pos, esc );
                        ++reps;
                        pos = ins.find_first_of( to_esc, pos + esc.size() + 1 );
                }
                return reps;
        }


        const entity_map &
        default_escapes_translations()
        {
//                 typedef ::Detail::phoenix::Detail::phoenix<entity_map,
//                         strtool_sharing_context<entity_map>,
//                         default_escapes_initializer
//                         > TMap;
		static entity_map bob;
		if( bob.empty() )
		{
			default_escapes_initializer()( bob );
		}
                return bob;
        }


        std::size_t strip_slashes( std::string &str, const char slash )
        {
                std::string::size_type osz;
                if( str.empty() || ((osz = str.size()) < 2 ) ) return 0;
                std::string::size_type pos = 0;
                std::size_t count = 0;
                pos = str.find( slash );
                if( pos == str.npos ) return 0;
                if( osz < 2 ) return 0;

                // strip escaped newlines. this is in a separate loop
                // because it used to be in a different function. :/
                pos = osz-2;
                std::string::size_type search;
                while( pos > 2 )
                {
                        char c = str[pos];
                        if( slash == c && (str[pos-1] != slash) )
                        {
                                ++count;
                                search = str.find_first_not_of( " \t\n", pos +1 );
                                if( search > pos + 1 )
                                {
                                        //COUT << "stripping until " << search << ":["<<str.substr(pos,search-pos)<<"]"<<endl;
                                        str.erase( pos, search -pos );
                                }
                                --pos; // one extra.
                        }
                        --pos;
                }
                
                pos = str.find( slash );
                while( !( (pos == std::string::npos) || (pos > str.size()-2) ) )
                {
                        // todo: search from the end, going
                        // backwards. This "might" be faster in terms
                        // of string's required workload.
                        ++count;
                        str.erase( pos, 1 );
                        if( slash != str[pos+1] )
                        {
                                pos += 1;
                        } else pos += 2;
                        pos = str.find( slash, pos );
                }

                return count;
        }

        void normalize_string( std::string &str )
        {
                //COUT << "normalize_string("<<str<<")" << endl;
                trim_string( str );
                strip_slashes( str );
                const char ch = str[0]; 
                if( '"' == ch || '\'' == ch )
                {
                        str.erase( 0, 1 );
                        str.resize( str.size() - 1 );
                }
                //COUT << "normalize_string("<<str<<")" << endl;
                return;
        }



        std::string trim_string( const std::string &str, TrimPolicy policy )
        {
                std::string foo = str;
                trim_string( foo, policy );
                return foo;
        }

        std::size_t trim_string( std::string &str, TrimPolicy policy )
        {
                if( str.empty() ) return 0;
                static const std::string space(" \t\n\r");
                std::size_t sc = 0;
                if( policy & TrimTrailing )
                {
                        while( !str.empty() && str.find_last_of( space ) == (str.size()-1) )
                        {
                                str.erase( str.size()-1 );
                                ++sc;
                        }
                }
                if( policy & TrimLeading )
                {
                        while( !str.empty() && (str.find_first_of( space ) == 0 ) )
                        {
                                str.erase( 0, 1 );
                                ++sc;
                        }
                }
                //CERR << "trim_string(...,"<<std::hex<<policy<<") = ["<<str<<"]"<<std::endl;
                return sc;
        }

        std::string
        first_token( const std::string & input )
        {
                if( input.empty() ) return input;
                return input.substr( 0, input.find_first_of( " \n\t" ) );
        }

        std::string
        after_first_token( const std::string & input )
        {
                if( input.empty() ) return input;
                std::string::size_type pos = input.find_first_of( " \n\t" );
                if( pos == std::string::npos ) return std::string();
                return trim_string( input.substr( pos ) );
        }


        int int4hexchar( char c )
        {
                int i = -1;
                if( c >= 48  && c <=57 ) // 0-9
                {
                        i = ((int)c - 48);
                }
                else if( c >= 65  && c <=70 ) // A-F
                {
                        i = ((int)c - 65) + 10;
                }
                else if( c >= 97  && c <=102 ) // a-f
                {
                        i = ((int)c - 97) + 10;
                }

                return i;
        }

        int hex2int( const std::string & wd )
        {
                unsigned int mult = 1;
                int ret = 0;
                char c;
                for( std::string::size_type i = wd.size(); i > 0; --i )
                {
                        //COUT << "i="<<i<<endl;
                        c = wd[i-1];
                        if( '#' == c ) continue;
                        ret += mult * int4hexchar( c );
                        mult = mult * 16;
                }
                return ret;
        }

	void default_escapes_initializer::operator()( entity_map & map )
	{
		map["\\"] = "\\\\";
		map["\'"] = "\\\'";
		map["\""] = "\\\"";
	}

        std::string
        expand_dollar_refs( const std::string & text, const entity_map & src )
        {
                std::string foo = text;
                expand_dollar_refs_inline( foo, src );
                return foo;
        }

        std::size_t
        expand_dollar_refs_inline( std::string & buffer, const entity_map & src )
        {
		using std::string;
                //CERR << "environment::expand_vars(["<<buffer<<"])"<<std::endl;
		if( buffer.size() < 2 ) return 0;

                string::size_type pos_a = 0, pos_b = 0;
                static const char vardelim = '$';
                pos_a = buffer.find( vardelim );
                if( string::npos == pos_a )
                {
                        return 0;
                }
                static const char opener = '{';
                static const char closer = '}';
                string tmpvar;
                std::size_t count = 0;
                static const string allowable_chars =
                        "abcdefghijklmnopqrstuvwxyz_a_bCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_?";
		// ^^^^^^ why is '?' in the list? So that eshell can support the shell-ish $? var :/.
                char atc;
                pos_a = buffer.size() - 1;
                bool slashmode = false;
		entity_map::const_iterator srcit, srcet = src.end();
                for( ; pos_a != string::npos; pos_a-- )
                {
                        atc = buffer[pos_a];
                        if( atc != vardelim )
                        {
                                continue;
                        }
                        if( pos_a>0 && !slashmode && buffer[pos_a-1] == '\\' ) slashmode = true;
                        if( slashmode )
                        {
				// arguable:
				--pos_a;  // strip 1 slash
				buffer.erase( pos_a, 1 );
				// todo: only remove slashe preeding $.
				// end arguable
                                slashmode = false;
                                continue;
                        }
                        pos_b = buffer.find_first_not_of( allowable_chars, pos_a+1 ); // find first non-variablename char
                        if( pos_b != pos_a +1 ) pos_b -= 1;
                        if( pos_b == string::npos )
                        {
                                pos_b = buffer.size() -1;
                        }

                        tmpvar.clear();
                        if( pos_b == pos_a + 1 ) // ${VAR} or $F, hopefully
                        {
                                atc = buffer[pos_b];
                                if( atc != opener )
                                {
                                        // $NONBRACED_VAR
                                        pos_b = buffer.find_first_not_of( allowable_chars, pos_b );
                                        tmpvar = buffer.substr( pos_a + 1, pos_b  );
                                        //CERR << "nonbraced var? ["<<tmpvar<<"]\n";
					//                                         tmpvar += atc;
					//                                         if( tmpvar.find_first_of( allowable_chars ) != 0 )
					//                                         {
					//                                                 tmpvar = string();
					//                                         }

                                }
                                else // ${VAR}
                                {
                                        //cout << "Activating Bat-parser! atc="<<atc<<" pos_a="<<pos_a<<" pos_b="<<pos_b << endl;
                                        const std::size_t maxpos = buffer.size()-1;
                                        while( atc != closer && pos_b <= maxpos )
                                        { // extract variable-name part:
                                                atc = buffer[++pos_b];
                                                if ( atc != closer ) tmpvar += atc;
                                        }
                                }
                        }
                        else
                        {
                                // extract variable-name part:
                                tmpvar = buffer.substr( pos_a+1 /*skip '$'*/, pos_b-pos_a );
                        }
                        //CERR << "expand_vars(): tmpvar=["<<tmpvar<<"]"<<endl;
                        if( tmpvar.empty() ) continue;
			srcit = src.find( tmpvar ); // don't expand unknown vars to empty strings.
			if( srcet == srcit ) continue;
                        tmpvar = (*srcit).second;
                        //CERR << "expand_vars(): expanded tmpvar=["<<tmpvar<<"]"<<endl;
                        ++count;
                        buffer.erase( pos_a, pos_b - pos_a +1 );
                        buffer.insert( pos_a, tmpvar.c_str() );
                }
                return count;
        }


} } } // namespaces

