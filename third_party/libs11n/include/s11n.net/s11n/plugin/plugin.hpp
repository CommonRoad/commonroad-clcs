#ifndef plugin_PLUGIN_HPP_INCLUDED
#define plugin_PLUGIN_HPP_INCLUDED 1

#include <s11n.net/s11n/plugin/path_finder.hpp>
#include <s11n.net/s11n/plugin/plugin_config.hpp>

//Added by Damien to make Windows compile work
//#include <s11n.net/s11n/export.hpp> // S11N_EXPORT_API
// ^^^ removed by stephan because it breaks the build tree on Unix
// platforms because the plugins dir must be built before the s11n
// dir, but export.hpp is not put in place until the s11n subdir is
// built. The reason for that structure is to cause circular deps like
// this one to explicitly break the build, so at least it did what it
// was supposed to. :)
#if !defined(S11N_EXPORT_API)
#ifdef WIN32
#  ifdef S11N_EXPORTS
#    define S11N_EXPORT_API __declspec(dllexport)
#  else
#    define S11N_EXPORT_API __declspec(dllimport)
#  endif
#else
#    define S11N_EXPORT_API
#endif
#endif

namespace s11n {
	/**
	   The plugin namespace encapsulates the braindeadly basic API
	   for loading plugins.

	   This namespace is an optional component of the s11n framework,
	   as it requires some platform-specific calls to open DLLs.

	   Some behaviours in this API are defined as "platform
	   specific", essentially meaning a combination of two things:

	   a) Your operating system.

	   b) The DLL loader this API is configured to use.

	   If you don't know which DLL loader is built with your copy
	   of this library, there are ways to find out:
		   
	   a) Under Win32, plugins using the native LoadModule()
	   function, and are thus always enabled. Win32 plugins are,
	   however, currently completely untested.

	   b) Unix: look in plugin_config.hpp. If
	   s11n_CONFIG_HAVE_LIBLTDL is set to a true value (not zero)
	   then this library is built with libltdl. If
	   s11n_CONFIG_HAVE_LIBLTDL is true, libdl is used (libltdl
	   trumps libdl). If neither are true, your tree was built
	   without plugins support.  Using ldd to find out which DLL
	   lib is used is not really accurate because libdl or libltdl
	   might be linked in to your app for reasons unrelated to
	   this library.

	*/
	namespace plugin
	{
		/**
		   The shared lookup path for plugins. It is
		   initialized to contain a path and list of
		   extensions defined in plugin_config.hpp.
		*/
		path_finder & path();

		/**
		   Returns path().find( basename ).
		*/
		//Added by Damien to make Windows compile work
      S11N_EXPORT_API std::string find( const std::string & basename );

		/**
		   Uses find(basename) to search for a file and, if
		   successful, opens it using a platforms-specific DLL
		   opener.  It does not manipulate the DLL except to
		   open it. The handle to the DLL is lost: it will be
		   closed by the OS when the application exits. (The
		   reasons for this are documented in the paper
		   entitled "Classloading in C++", available from
		   http://s11n.net/papers/.)

		   Returns the path to the DLL, or an empty string on
		   error. There is currently no way of knowing what
		   the error might have been.

		   The path() is used to find the DLL. If basename is
		   an absolute path to an existing file, it will be
		   used as-is, which means that a client-specified
		   path_finder object may be used to find DLLs, and
		   then pass them to this function to do the
		   platform-specific work of opening the DLL.
		*/
		//Added by Damien to make Windows compile work
      S11N_EXPORT_API std::string open( const std::string & basename );

		/**
		   Called immediately after open() fails, it returns a
		   platforms-specific error string.

		   On platforms using libdl or libltdl, this function
		   returns the result of calling dlerror() or
		   lt_dlerror(), respectively.  On Windows platforms
		   it always returns an empty string (anyone now how
		   to get such error strings from Windows?).

		   Calling this twice in a row will, without calling
		   open() in between always cause an empty string to
		   be returned on the second call. See the man pages
		   for dlerror() or lt_dlerror() for more details on
		   this, though this function also behaves that way
		   under Win32.

		*/
		//Added by Damien to make Windows compile work
      S11N_EXPORT_API std::string dll_error();

	} // namespace plugin
} // namespace s11n


#endif // plugin_PLUGIN_HPP_INCLUDED
