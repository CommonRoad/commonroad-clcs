////////////////////////////////////////////////////////////////////////
// platform-specific parts of plugin.{c,h}pp: for libltdl and libdl
////////////////////////////////////////////////////////////////////////
#if s11n_CONFIG_HAVE_LIBLTDL // prefer libltdl, but if it's not available...
#  include <ltdl.h>
#else           // assume libdl is.
#  include <dlfcn.h> // this actually has a different name on some platforms!
#endif // s11n_CONFIG_HAVE_LIBLTDL

#include <iostream>
#ifndef CERR
#define CERR std::cerr << __FILE__ << ":" << std::dec << __LINE__ << " : "
#endif

namespace s11n { namespace plugin {

	static std::string m_dll_error; // internal holder for value returned by dll_error().

        std::string dll_error()
        {
		if( m_dll_error.empty() ) return m_dll_error;
		std::string ret = m_dll_error;
		m_dll_error = std::string();
		return ret;
        }

	std::string open( const std::string & basename )
	{
		std::string where = find( basename );

		if( where.empty() )
		{
			m_dll_error = std::string("s11n::plugin::open(")
				+ basename
				+ std::string( "): No DLL found.");
			// CERR << m_dll_error << "\n";
			return std::string();
		}

                static bool donethat = false;
                if( !donethat && (donethat=true) )
                {
                        // About these dlopen(0) calls:
                        // They open the main() app and are
                        // required. If they are not called at some
                        // point then loading DLLs will not work.
#if s11n_CONFIG_HAVE_LIBLTDL
                        lt_dlinit();
                        lt_dlopen( 0 );
#else
                        dlopen( 0, RTLD_NOW | RTLD_GLOBAL );
#endif
                }

                void * soh = 0; // DLL handle

		//CERR << "s11n::plugin::open("<<basename<<"): trying to open: " << where << "\n";

#if s11n_CONFIG_HAVE_LIBLTDL // libltdl:
                soh = lt_dlopen( where.c_str() );
#else // libdl:
                soh = dlopen( where.c_str(), RTLD_NOW | RTLD_GLOBAL );
#endif
                if( 0 == soh )
                {
			const char * err = 0;
#if s11n_CONFIG_HAVE_LIBLTDL // libltdl:
			err = lt_dlerror();
#else // libdl:
			err = dlerror();
#endif
			m_dll_error = err ? std::string(err) : std::string("unknown error");
			if( 0 ) CERR << "s11n::plugin::open("<<basename<<"): Error opening DLL:\n"
				     << "file="<<where<<"\n"
				     << "error="<<m_dll_error
				     << "\n";
                        return std::string();
                }
		else
		{
			m_dll_error = std::string();
		}
		return where;
	}

}} // namespace

