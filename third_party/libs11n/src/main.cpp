////////////////////////////////////////////////////////////////////////////////
// main_dn.cpp
// Implmentation for s11nconvert utility, to convert data between the various
// Serializers.
// Author: stephan@s11n.net
// License: Public Domain
////////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <string>
#include <memory> // auto_ptr

#ifndef WIN32
#  include <unistd.h> // isatty()
#endif // !WIN32

#include <s11n.net/s11n/s11n_debuggering_macros.hpp> // CERR
#include <s11n.net/s11n/s11nlite.hpp> // s11nlite framework
#include <s11n.net/s11n/io/serializers.hpp> // utility code for s11n::io
#include <s11n.net/s11n/s11n_config.hpp> // s11n_CONFIG_HAVE_ZFSTREAM
#include "argv_parser.hpp"

#if s11n_CONFIG_HAVE_ZFSTREAM
#  include <s11n.net/zfstream/zfstream.hpp>
#endif

#if s11n_CONFIG_ENABLE_PLUGINS
#  include <s11n.net/s11n/plugin/plugin.hpp>
#endif

// DLLS_WORKIE is a temporary workaround for no DLL support in 1.1
#define DLLS_WORKIE (0)

bool s11nc_verbose = false;
#define VERBOSE if( s11nc_verbose ) CERR

void show_version_info()
{
	std::cout << s11n_S11N_LIBRARY_VERSION << "\n";
	if( s11nc_verbose) {
		std::cout << "Linked against s11n version " << s11n::library_version() << "\n"
			  << "s11nconvert is a tool to convert data between the formats supported by libs11n.\n"
			  << "License: " << s11n_PACKAGE_LICENSE << "\n"
			  << "This software comes with NO WARRANTY of any type.\n"
			  << "Find out more about s11n at http://s11n.net" << std::endl;
	}
}

void show_help()
{

	show_version_info();

	std::ostream & os = std::cout;
	os << "\nCommand-line options:\n\n";

#define HOP(A,H) os << "\t" << A << ":\n\t\t" << H << "\n";
	HOP("-?, --help", "Shows help text on stdout and exits with code 0." );
	HOP("-f,--file FILENAME", "Sets input file name. The name '-' means stdin." );
	HOP("-K, --known-serializers", "Listed known/loaded Serializers (format handlers).");
	HOP("-o, --output FILENAME", "Sets output filename. The name '-' means stdout." );
	HOP("-s, --format SERIALIZER", "Sets the output Serializer (i.e., output format)." );
	HOP("-S SERIALIZER", "Forces the specified input Serializer to be used (very rarely needed)." );
	HOP("-v, --verbose", "Enables verbose mode: additional info will go to stderr." );
	HOP("-V, --version", "Shows version number on stdout and exits with code 0. Use -v to output more info." );

#if s11n_CONFIG_ENABLE_PLUGINS
	HOP("-dl DLL", "Opens DLL, which may be an absolute path or a name findable using the s11n::plugin::path(). Causes app to exit with an error if DLL is not found or cannot be opened." );
	HOP("-DL DLL", "Identical to -dl except that a failed DLL load is not fatal." );
#endif


#if s11n_CONFIG_HAVE_ZFSTREAM
	HOP("-z","Compress output files with gzip. Works only on files, not stdout." );
	HOP("-bz","As for the -z option, but applies to bzip2 compression. -z and -bz may not be used together." );
#endif // zfstream?

#undef HOP

	os << "\nAll arguments may be in any of these forms:\n"
	   << "\t-ARG VALUE  (an empty value is assumed to be a boolean 1)\n"
	   << "\t-ARG=VALUE (quote the value if it contains spaces)\n"
	   << "The number of leading dashes for an argument is absolutely irrelevant. "
	   << "When an argument is specified more than once, only the last one set takes effect.\n";

	os << "\nSeveral options from s11nconvert 1.0.x are currently not available, "
	   << "but will be re-added eventually.\n"
	   << "\nYou can get more detailed help in the s11nconvert man page.\n";

	os << std::endl;

}


enum {
// Reminder: these entries have explicit values so that 
// we can publish them in the man page. If this enum
// changes, UPDATE THE MAN PAGE.
ErrorNoOutputSerializer = 1,
ErrorFileNotFound = 2,
ErrorNoInputFile = 3,
ErrorLoadNodeFailed = 4,
ErrorSaving = 5,
ErrorForcedSerializer = 6,
ErrorUsageError = 7,
ErrorNoOutputSerializers = 8,
ErrorException = 9
};

int
load_dlls( int argc, char **argv )
{
	if( argc < 2 ) return 0;
	std::string arg;
	std::string foo;
	int mode = 0; // 1 == tolerant, 2 == intolerant
	for( int i = 1; i < argc; ++i )
	{
		mode = 0;
		arg = argv[i];
		if( "-DL" == arg ) mode = 1; // tolerant
		else if( "-dl" == arg ) mode = 2; // intolerant
		else continue;

#if ! s11n_CONFIG_ENABLE_PLUGINS
		throw std::runtime_error( "-dl/-DL are not supported because your libs11n was built without plugins support." );
#else
		if( argc == ++i )
		{
			throw std::runtime_error( "-dl/-DL parameters require a following argument (filename or DLL class name)" );
		}
		arg = argv[i];
		foo = s11n::plugin::open( arg );
		if( foo.empty() )
		{
			std::ostringstream errmsg;
			errmsg << "DLL error: '"<<arg<<"': " << s11n::plugin::dll_error();
			if( 1 == mode )
			{
				CERR << "Warning: " << errmsg.str() << '\n';
				continue;
			}
			throw std::runtime_error( errmsg.str() );
		}
		VERBOSE << "Loaded DLL: " << foo << '\n';
#endif // s11n_CONFIG_ENABLE_PLUGINS
	}
	return 0;
}


int
do_everything( int argc, char **argv )
{
	s11n::Private::s11n_1_1_assertion(); // see the API docs for why this is here!

#define ERR std::cerr << "s11nconvert error: "

	cliutil::argv_map args;
	cliutil::unnamed_args_list unused;

	typedef s11nlite::node_traits NTR;
	s11nlite::node_type argn; // we're gonna cheat a bit and use this for arg parsing

	//int ArgC = 
		cliutil::parse_args( argc, argv, 1, args, unused );

	cliutil::argv_map::const_iterator amit = args.begin();
	for( ; args.end() != amit; ++amit )
	{
		NTR::set( argn, (*amit).first, (*amit).second );
	}

        if( NTR::is_set( argn, "v" ) || NTR::is_set( argn, "verbose" ) )
        {
                s11nc_verbose = true;
        }
        // VERBOSE << "Verbose mode on.\n";

	load_dlls( argc, argv ); // do this after verbosity check.



        ////////////////////////////////////////////////////////////////////////
        // Load list of known/registered Serializers.
        // Do this AFTER loading DLLs so that DLLs may install additional Serializers...
        std::string known_ser;
        typedef std::list< std::string > SerList;
        SerList serlist;
        s11n::io::serializer_list<s11nlite::node_type>( serlist, !s11nc_verbose );
        SerList::const_iterator cit = serlist.begin(),
                cet = serlist.end();
        for( ; cet != cit; ++cit )
        {
                known_ser += (*cit) + (s11nc_verbose ? "\n" : " ");
        }
        if( known_ser.empty() )
        {
                ERR << "No data formats loaded!\n"
		    << "This means we couldn't find any built-in Serializers, "
		    << "which is probably inidicitive of a problem in your local s11n installation.\n"
		    << "Make sure that your s11nconvert is not linking against an older, internally incompatible libs11n "
		    << "by using 'ldd s11nconvert'.\n"
			;
                return ErrorNoOutputSerializers;
        }


        if( NTR::is_set( argn, "known-serializers" ) || NTR::is_set( argn, "K" ) )
        {
                std::cout << known_ser << std::endl;
                return 0;
        }
        ////////////////////////////////////////////////////////////////////////


        if( NTR::is_set( argn, "version" ) || NTR::is_set( argn, "V" ) )
        {
                show_version_info();
                return 0;
        }

        if( NTR::is_set( argn, "help" ) || NTR::is_set( argn, "?" ) )
        {
                show_help();
                return 0;
        }


#if s11n_CONFIG_HAVE_ZFSTREAM
        if( NTR::is_set( argn, "z" ) && NTR::is_set( argn, "bz" ) )
        {
                ERR << "-z and -bz may not be used together." << std::endl;
                return ErrorUsageError;
        }
#endif // s11n_CONFIG_HAVE_ZFSTREAM


        std::string fmt = NTR::get( argn, "s", NTR::get( argn, "format", std::string() ) );
        if( fmt.empty() )
        {
                ERR << "No output format (serializer) specified. "
                    << "Try using [-s or --format] with one of:\n"
                    << known_ser << "\n"
		    << "Use -? or --help to see the full list of options.\n";
                return ErrorNoOutputSerializer;
        }


         typedef std::unique_ptr<s11nlite::serializer_interface> APSER;
         APSER ser = APSER( s11nlite::create_serializer( fmt ) );
         if( ! ser.get() )
         {
                 ERR << "No Serializer found for name '"<<fmt<<"'. Try one of:\n"
                     << known_ser << "\n";
                 return ErrorNoOutputSerializer;
         }

 
         bool usegz = false;
         bool usebz = false;
         if( NTR::is_set( argn, "z" ) )
         {
                 usegz = true;
                 
         }
	 else if( NTR::is_set( argn, "bz" ) )
	 {
                 usebz = true;
		 
	 }

	 if( usegz || usebz )
	 {
#if s11n_CONFIG_HAVE_ZFSTREAM
		 zfstream::compression_policy( usegz ? zfstream::GZipCompression : zfstream::BZipCompression );
		 VERBOSE << "Using compression: " << (usegz ? "gzip" : "bzip2") << "\n";
#else
		 ERR << "zlib/bzip2 compression is not supported by your libs11n.\n";
		 return ErrorUsageError;
#endif
	 }


	 std::string ifname = NTR::get( argn, "f", NTR::get( argn, "file", std::string() ) );
         std::istream * is = 0;
         bool use_infile = false;
         if( ( "-" == ifname )
#ifndef WIN32
	     || ! isatty(STDIN_FILENO)
	     // TODO: i doubt windows has a way of checking this. If it doesn't, i want to
	     // exit with an error here, rather than risk incompatible behaviour.
#endif // !WIN32
	     )
         {
                 is = &std::cin;
         }
         else if( ifname.empty() )
         {
                 ERR << "No input file specified.\n";
                 return ErrorNoInputFile;
         }
         else
         {
                 use_infile = true;
         }



         std::ostream * os = 0;

	 std::string ofname = NTR::get( argn, "o", NTR::get(argn, "output", std::string() ) );
         bool use_ofile = false;
         if( ofname.empty() || ("-" == ofname) )
         {
                 if( usegz || usebz )
                 {
                         ERR << "Compression (-z and -bz) works only for files, not cout." << std::endl;
                         return ErrorUsageError;
                 }
                 os = &std::cout;
         }
         else
         {
                 use_ofile = true;
         }

         typedef std::unique_ptr<s11nlite::node_type> NAP;
         NAP innode;
         std::string forcedinser = NTR::get( argn, "S", std::string() );
         if( ! forcedinser.empty() )
         {
                 VERBOSE << "Forcing input serializer: " << forcedinser << "\n";
                 s11nlite::serializer_interface * forceser = s11nlite::create_serializer( forcedinser );
                 if( 0 == forceser )
                 {
                         ERR << "Could not load forced input serializer '"<<forcedinser<<"'!\n";
                         return ErrorForcedSerializer;
                 }
                 innode = NAP( use_infile
                               ? forceser->deserialize( ifname )
                               : forceser->deserialize( *is ) );
         }
         else
         {
                 innode = NAP( use_infile ? s11nlite::load_node( ifname ) : s11nlite::load_node( *is ) );
         }
         if( ! innode.get() )
         {
                 ERR << "Error loading node tree from input stream!\n";
                 return ErrorLoadNodeFailed;
         }

        // And FINALLY... we get down to the real work...
        // these couple lines almost now don't seem worth all the hassle ;)

	 bool workie = use_ofile ? ser->serialize( *innode, ofname ) : ser->serialize( *innode, *os );
	 if( ! workie )
	 {
		 ERR << "Error saving output!\n";
		 return ErrorSaving;
	 }

	return 0;
#undef ERR
};


int
main( int argc, char **argv )
{
	try
	{
		return do_everything( argc, argv );
	}
	catch( std::exception const & ex )
	{
		CERR << "Exception caught: " << ex.what() << '\n';
		return ErrorException;
	}
	return 0;
}
