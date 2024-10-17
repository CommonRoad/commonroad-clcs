//    Author: stephan beal <stephan@s11n.net>
//    License: Public Domain

#include <iostream>
// #include <stdlib.h>		// getenv()

#include <s11n.net/s11n/plugin/path_finder.hpp>
// #include <s11n.net/s11n/s11n_debuggering_macros.hpp> // CERR

#ifndef WIN32
#  define WIN32 0
#endif

#if WIN32
#  include  <io.h>
#  include  <stdio.h>
#else
#  include <unistd.h>
#endif



namespace s11n { namespace plugin {


        path_finder::~path_finder()
        {
                // DTOROUT(path_finder) << this->path_string() << std::endl;
        }

	path_finder::path_finder( const std::string & p, const std::string & e, const std::string & pathsep )
	{
		this->path_separator( pathsep );
		this->path( p );
		this->extensions( e );
	}


	bool path_finder::empty() const
	{
		return this->paths.empty() && this->exts.empty();
	}

	const std::string & path_finder::path_separator() const
	{
		return ( ( path_finder * ) this )->pathseparator;
	}

	void path_finder::path_separator( const std::string & sep )
	{
		this->pathseparator = sep;
	}

        std::string path_finder::join_list( const string_list & list, const std::string & separator )
        {
                std::string ret;
                unsigned long count = list.size();
                unsigned long at = 0;
                string_list::const_iterator it = list.begin();
                string_list::const_iterator et = list.end();
                for(; it != et; ++it )
                {

                        ret += (*it);
                        if( ++at != count ) ret += separator;
                }
                return ret;
        }
	std::string path_finder::path_string() const
	{
                return this->join_list( this->paths, this->pathseparator );
	}

	const path_finder::string_list & path_finder::path() const
	{
		return this->paths;
	}

	path_finder::string_list & path_finder::path()
	{
		return this->paths;
	}

	std::string path_finder::extensions_string() const
	{
                return this->join_list( this->exts, this->pathseparator );
	}

	const path_finder::string_list & path_finder::extensions() const
	{
		return this->exts;
	}

	path_finder::string_list & path_finder::extensions()
	{
		return this->exts;
	}


        std::size_t tokenize_to_list( const std::string & str, std::list<std::string> & li, const std::string & sep )
        { // internal helper function
                if( str.empty() ) return 0;

                std::size_t c = 0;

                std::string token;
                std::string::size_type sz = str.size();
                for( std::string::size_type i = 0; i < sz; i++ )
                {
                        if( sz-1 == i ) token += str[i];
                        if( str.find( sep, i ) == i || (sz-1 == i) )
                        {
                                //CERR << "token="<<token<<std::endl;
                                li.push_back( token );
                                token = "";
                                i += sep.size() - 1;
                                continue;
                        }
                        token += str[i];
                }
		return c;
        }

	std::size_t path_finder::path( const std::string & p )
	{
                this->paths.erase( this->paths.begin(), this->paths.end() );
                return tokenize_to_list( p, this->paths, this->pathseparator );
	}

	std::size_t path_finder::path( const path_finder::string_list & p )
	{
		this->paths = p;
		return this->paths.size();
	}

	void path_finder::add_path( const std::string & p )
	{
                tokenize_to_list( p, this->paths, this->pathseparator );
	}


	std::size_t path_finder::extensions( const std::string & p )
	{
                this->exts.erase( this->exts.begin(), this->exts.end() );
                return tokenize_to_list( p, this->exts, this->pathseparator );
	}

	std::size_t path_finder::extensions( const path_finder::string_list & e )
	{
		this->exts = e;
		return this->exts.size();
	}

	void path_finder::add_extension( const std::string & p )
	{
                tokenize_to_list( p, this->exts, this->pathseparator );
	}

	// static
	bool path_finder::is_accessible( const std::string & path )
	{
#if WIN32
#  define CHECKACCESS _access
#  define CHECKRIGHTS 0
#else
#  define CHECKACCESS access
#  define CHECKRIGHTS F_OK
#endif

                return 0 == CHECKACCESS( path.c_str(), CHECKRIGHTS );
#undef CHECKACCESS
#undef CHECKRIGHTS
	}

	std::string path_finder::basename( const std::string & name )
	{
		std::string::size_type slashat = name.find_last_of( path_finder::dir_separator() );
		if ( slashat == std::string::npos )
			return name;
		return name.substr( slashat + 1 );
	}


		std::string path_finder::dir_separator()
		{
#if WIN32
			return std::string( "\\" );
#else
			return std::string( "/" );
#endif
		}


	std::string path_finder::find( const std::string & resource, bool check_cache ) const
	{
		//static const std::string NOT_FOUND = "path_finder::find() : no findie";
		if( resource.empty() ) return resource;

#define CHECKPATH(CHECKAT)  \
        if( ! CHECKAT.empty() && path_finder::is_accessible( CHECKAT ) ) \
        { this->hitcache[resource] = CHECKAT; return CHECKAT; }

		//CERR << "find( " << resource << " )" << std::endl;
                if( check_cache )
                {
                        std::map <std::string,std::string>::iterator mapiter;
                        mapiter = this->hitcache.find( resource );
                        if( this->hitcache.end() != mapiter ) return (*mapiter).second;
                }

                CHECKPATH( resource );

                string_list::const_iterator piter = this->paths.begin();
                string_list::const_iterator eiter = this->exts.begin();

		std::string path;
		std::string ext;

		if ( path_finder::is_accessible( resource ) )
                        return resource;

                piter = this->paths.begin();
		std::string checkhere;
		while ( piter != this->paths.end() )
		{
			path = ( *piter );
			if ( !path.empty() )
			{
				path += path_finder::dir_separator();
			}
			++piter;
			checkhere = path + resource;
			//CERR << "find( " << resource << " ) checking " << checkhere << std::endl;
			CHECKPATH( checkhere );
			eiter = this->exts.begin();
			while ( eiter != this->exts.end() )
			{
				ext = ( *eiter );
				++eiter;
				checkhere = path + resource + ext;
				//CERR << "find( " << resource << " ) checking " << checkhere << std::endl;
				CHECKPATH( checkhere );
			}
		}
		//CERR << "find( "<<resource<<" ): not found :(" << std::endl;
                // so arguable:
		// this->hitcache[resource] = "";
		return std::string();
	}
#undef CHECKPATH
        void path_finder::clear_cache()
        {
                this->hitcache.clear();
        }
//         /**
//            bin_path_finder is a path_finder which uses the environment's PATH by default.
//         */
// 	class bin_path_finder:public path_finder
// 	{
// 	      public:
// 		bin_path_finder();
// 		virtual ~ bin_path_finder();
// 	};

//         bin_path_finder::bin_path_finder() : path_finder( ::getenv( "PATH" ) )
//         {
//                 this->extensions( ".sh" );
// #if CONFIG_HAVE_CYGWIN
//                 this->extensions(".exe:.bat");
// #endif
//         }
//         bin_path_finder::~bin_path_finder()
//         {
//         }


} }				// namespace

