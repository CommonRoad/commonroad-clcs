#ifndef s11n_net_S11N_PLUGIN_PATHFINDER_HPP_INCLUDED
#define s11n_net_S11N_PLUGIN_PATHFINDER_HPP_INCLUDED
// Author: stephan beal <stephan@s11n.net>
// License: Public Domain



#include <string>
#include <list>
#include <map>
#include <iostream>

namespace s11n { namespace plugin {
        /**
           path_finder searches for keys using a set of prefixes
           (paths) and suffixes (file extensions).
           
           Example:

           <pre>
           path_finder p;
           p.path( "/lib:/usr/lib" );
           p.extensions( ".a:.so" );
           std::cout << p.find( "libz" ) << std::endl;
           </pre>

           That would print an empty string if it finds nothing, or a
           string if it finds any of the following:

           - libz (that is, if the value passed is an existing file,
                   it is returned as-is).
           - /lib/libz
           - /lib/libz.a
           - /lib/libz.so
           - /usr/lib/libz
           - /usr/lib/libz.a
           - /usr/lib/libz.so


	   Maintainer's note:

	   This cide was one of my very first STL-based classes, and
	   the implementation probably shows that very clearly. That
	   said, it has worked well for me for some four years now
	   without any appeciable maintenance. :)
         */
	class path_finder
	{
	      public:

                /**
                   A list type returned by some functions.
                 */
                typedef std::list<std::string> string_list;


		/**
                   Creates object with the given path/extension list.
                */
                path_finder( const std::string & path = std::string(), const std::string & ext = std::string(), const std::string & pathsep = ":" );

                virtual ~path_finder();
                
		/**
                   Returns a ":"-separated string of all paths added via add/path().
                */
		  std::string path_string() const;

                /**
                   Sets the string used as a separator for the
                   string-based variants of path(), extentions(), etc.
                */
		void path_separator( const std::string & sep );

                /**
                   Returns the path separator string. Default is ":";
                 */
		const std::string & path_separator() const;

		/**
                   Sets the path to p, which should be a path_separator()-delimited string.
                   Returns the number of path elements parsed from p.
                */
		virtual std::size_t path( const std::string & p );

                /**
                   Sets the path to the given list of directories.
                   Returns the number of elements in the list.
                 */
		virtual std::size_t path( const string_list & p );

		/**
                   Adds p to the path. May be path_separtor()-delimited.
                */
		virtual void add_path( const std::string & p );

		/**
                   Adds a "search extension." Sample:
                   finder.extension( ".txt:.TXT" ); Will now try all
                   path combinations with the rightmost characters
                   matching ".txt" or ".TXT" (in that order). Be sure
                   to include a period if you want that searched -
                   that is so this class can be used to find non-files
                   and those with non-traditional extensions, like
                   "foo_EXT".
                */
		virtual void add_extension( const std::string & ext = std::string() );
		/**
                   like add_extension(), but overwrites extension list.
                   Returns the number of entries parsed from the string.
                */
		virtual std::size_t extensions( const std::string & ext );
                /**
                   Sets the extensions list to the given list.
                   Returns the number of entries in p.
                 */
		virtual std::size_t extensions( const string_list & p );

		/**
                   Returns the path_separator()-delimited listed of file
                   suffixes to use when searching for a path.
                */
		  std::string extensions_string() const;
                /**
                   Returns this object's extensions list.
                 */
		const string_list & extensions() const;

		/** Non-const overload, intended for serialization. */
		string_list & extensions();

                /**
                   Helper function to collapse a list into a string.

		   This function was changed from a normal member to
		   static member in s11n version 1.1.3.
                 */
                static std::string join_list( const string_list & list, const std::string & separator );

                /**
                   Returns true if path is readable.
                 */
		static bool is_accessible( const std::string & path );

		/**
                   Returns the "base name" of the given string: any part
                   following the final directory separator character.
                */
		static std::string basename( const std::string & );

                /**
                   Returns a platform-dependent path separator. This
                   is set when the class is compiled.
                 */
		static std::string dir_separator();

		/**
                   Returns the full path of the given resource,
                   provided it could be found using the available
                   lookup paths/extensions and is readable.  Note that
                   this might return a relative path, especially if
                   the resourcename passed to it immediately resolves
                   to an existing resource.  It returns an empty
                   string if the resourcename cannot be found in the
                   filesystem tree (or is otherwise unaccessible).

                   If check_cache is false then this function ignores
                   its lookup cache and searches again, otherwise it
                   uses a cache. When caching it will always return
                   the same result for any given resourcename.
                */
		std::string find( const std::string & resourcename, bool check_cache = true ) const;

                /**
                   Empties the hit-cache used by find().
                */
                void clear_cache();

		/**
                   Returns a list of all items added via add_path() and path().
                */
		const string_list & path() const;
		/** Non-const overload, intended for serialization. */
		string_list & path();


		/** Returns true if this object has no paths or extensions. */
		bool empty() const;

	      private:
		string_list paths;
		string_list exts;
		std::string pathseparator;
		typedef std::map < std::string, std::string > StringStringMap;
		typedef StringStringMap::iterator StringStringIterator;
		mutable StringStringMap hitcache;
	};


}}				// namespace

#endif // s11n_net_S11N_PLUGIN_PATHFINDER_HPP_INCLUDED
