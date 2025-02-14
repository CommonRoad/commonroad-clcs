#include <s11n.net/s11n/plugin/plugin.hpp>
#include <s11n.net/s11n/plugin/plugin_config.hpp>

#include <iostream>
#ifndef CERR
#define CERR std::cerr << __FILE__ << ":" << std::dec << __LINE__ << " : "
#endif

namespace s11n { namespace plugin {

// 	/** Internal marker type. */
// 	struct default_path_context {};

// 	struct path_initer
// 	{
// 		void operator()( path_finder & p )
// 		{
// 			p.add_path( s11n_CONFIG_PLUGINS_PATH );
// 			p.add_extension( s11n_CONFIG_DLL_EXTENSION );
// 		}
// 	};

	path_finder & path()
	{
		//return ::s11n::Detail::phoenix<path_finder,default_path_context,path_initer>::instance();
		static path_finder bob;
		static bool donethat = false;
		if( (!donethat) && (donethat=true) )
		{
			bob.add_path( s11n_CONFIG_PLUGINS_PATH );
			bob.add_extension( s11n_CONFIG_DLL_EXTENSION );
// 			path_initer()( bob );
		}
		return bob;
	}

	std::string find( const std::string & basename )
	{
		// CERR << "find("<<basename<<")... path="<<path().path_string()<<"\nextensions="<<path().extensions_string()<<"\n";
		return path().find( basename );
	}

} } // namespace


#if !defined(WIN32) || !(WIN32)
#  if s11n_CONFIG_HAVE_LIBLTDL || s11n_CONFIG_HAVE_LIBDL
#    include "plugin.dl.cpp"
#  else
#    include "plugin.noop.cpp"
#  endif // libdl/ltdl?
#else
#    include "plugin.win32.cpp"
#endif // WIN32?
