// Author: stephan@s11n.net
// License: Public Domain

#include <iostream>
#include <string>
#include <vector>
#include <map>

#ifndef CERR
#define CERR std::cerr << __FILE__ << ":" << std::dec << __LINE__ << " : "
#endif
#include <iostream>

#include "argv_parser.hpp"

namespace cliutil {

        int
        parse_args( int argc, char ** argv, int startAtPos, argv_map & target, unnamed_args_list & target2 )
        {
                using namespace std;
                if( startAtPos >= argc ) return -1;
                std::string argprefix = "-";
                int acount = 0;
                std::string v;
                std::string a;
		std::string key;
                std::string nextarg;
                static const string numbers = "0123456789";
                bool skipnext = false;
		std::string::size_type eqpos = std::string::npos;
		std::string::size_type bogopos = std::string::npos;
                for( int i = startAtPos; i < argc; i++ )
                {
                        if( skipnext ) { skipnext = false; continue; }
                        a = std::string( (const char *)argv[i] );
			if( "--" == a ) break; // conventional "end of arguments" argument
                        if( a.find(argprefix)!=0 )
			{
				target2.push_back( a );
				continue;
			}
                        v = std::string();
                        ++acount;
			bogopos = a.find_first_not_of( "-" );
			eqpos = a.find('=');
			key = a.substr( bogopos, eqpos-bogopos );
			if( std::string::npos != eqpos )
                        {

				if( eqpos != (a.size()-1) )
				{
					v = a.substr( eqpos+1, a.size()-eqpos );
				}
				else
				{
					v = "";
				}
                        }
                        else // else it's space-separated or a boolean flag
                        {
                                if( i < argc-1 )
                                {
                                        nextarg = std::string(argv[i+1]);
                                        if( nextarg == "-" )
                                        {
                                                // workaround for things like: myapp --infile -
                                                v = "-";
                                                skipnext = true;
                                        }
                                        else if( nextarg.find(argprefix) == 0 )
                                        {
                                                if( nextarg.find_first_of( numbers ) == 1 )
                                                { // let's assume it's a negative number, not a flag
                                                        // todo: actual atol() or atod() check
                                                        // Note that this logic means that numbers as flags is not supported. :/
                                                        skipnext = true;
                                                        v = nextarg;
                                                }
                                                else
                                                { // nextarg is argprefix'd, so treat this as a boolean flag
                                                        v = "1";
                                                }
                                        }
                                        else
                                        { // it was space-separated: --foo bar
                                                v = nextarg;
                                                skipnext = true;
                                        }
                                }
                                else
                                {
                                        v = "1"; // the final item is an argument, toggling it to true.
                                }
                        }
                        target[key] = v;
                        // CERR << "["<<key<<"] = ["<<v<<"]"<<endl;
                }
                return acount;
        }

} // namespace acme

#if ARGV_PARSER_TEST_MAIN
int main( int argc, char ** argv )
{
	acme::argv_map av;
	acme::unnamed_args_list unnamed;
	int ac = acme::parse_args( argc, argv, 1, av, unnamed );

	CERR << "Arg count == " << ac << "\n";
	CERR << "Arg av.size() == " << av.size() << "\n";
	CERR << "Arg unnamed.size() == " << unnamed.size() << "\n";

}
#endif // ARGV_PARSER_TEST_MAIN
