#ifndef s11n_net_s11n_STRINGTOOL_HPP_INCLUDED
#define s11n_net_s11n_STRINGTOOL_HPP_INCLUDED 1

#include <string>
#include <map>
#include <locale>
#include <iostream>
#include <sstream>

namespace s11n { namespace io {
/**
The strtool namespace encapsulates a set of utility functions for
working with string objects. This mini-lib has unfortunately followed
me from source tree to source tree like a little virus. While i have
no special love for this code, it has proven useful time and time again.
*/
namespace strtool {

        /**
           The functions in the Private namespace should not be used
           by client code.
        */
        namespace STPrivate
        {


                /**
                   Lexically casts str to a value_type, returning
                   errorVal if the conversion fails.

                   TODO: implement the following suggestion from  
                   Kai Unger <kai.unger@hacon.de> (21 Sept 2004):

                   When the cast is done, you should check if there
                   are unread characters left. For example, casting
                   "1.2this_definitly_is_not_a_number" to double will
                   not result in returning the error value, because
                   conversion of "1.2" to 1.2d succeeds and the rest
                   of the string is ignored.
                */
                template <typename value_type>
                value_type from_string( const std::string & str, const value_type & errorVal ) throw()
                {
                        std::istringstream is( str );
                        if ( !is )
                                return errorVal;
                        value_type foo = value_type();
                        if ( is >> foo )
                                return foo;
                        return errorVal;
                }

                /**
                   Returns a string representation of the given
                   object, which must be ostreamble.
                */
                template <typename value_type>
                std::string to_string( const value_type & obj ) throw()
                {
                        std::ostringstream os;
                        // os << std::fixed;
                        os << obj;
                        return os.str();
                }

//                 inline std::string to_string( double d ) throw()
//                 {
//                         std::ostringstream os;
//                         os << std::fixed << d;
//                         return os.str();
//                 }

                /**
                   Convenience/efficiency overload.
                */
                inline std::string from_string( const std::string & str, const std::string & /*errorVal*/ ) throw()
                {
                        return str;
                }

                /**
                   Convenience/efficiency overload.
                */
                inline std::string from_string( const char *str, const char * /*errorVal*/ ) throw()
                {
                        return str;
                }

                /**
                   Convenience/efficiency overload.
                */
                inline std::string to_string( const char *obj ) throw()
                {
                        return obj ? obj : "";
                }

                /**
                   Convenience/efficiency overload.
                */
                inline std::string to_string( const std::string & obj ) throw()
                {
                        return obj;
                }


        } // end STPrivate namespace

        /**
           Convenience typedef for use with translate_entities().
         */
        typedef std::map<std::string,std::string> entity_map;

        /**
           For each entry in the input string, the characters are
           mapped to string sequences using the given
           translation_map. Where no mappings exist, the input
           sequence is left as-is.

           It returns the number of translations made.

           If reverse_translation == true then a reverse mapping is
           done: map values are treated as keys.

           This is useful, for example, for doing XML-entity-to-char
           conversions.

	   Complexity is essentially linear, based on a combination of
	   buffer.size() and translation_map.size(). Best used with
	   small maps on short strings! The speed can be increased
	   signifcantly, but probably only if we restrict keys and
	   values to 1 character each.

	   Design note: this really should be a function template,
	   accepting any lexically-castable key/val types, but the
	   function is quite long, and therefore not really suitable
	   for inclusion in the header.
        */
        std::size_t translate_entities( std::string & buffer, const entity_map & translation_map, bool reverse_translation = false );


        /**
           A policy enum used by trim_string().
        */
        enum TrimPolicy {
        /**
           Trim only leading spaces.
         */
        TrimLeading = 0x01,
        /**
           Trim only trailing spaces.
         */
        TrimTrailing = 0x02,
        /**
           Trim leading and trailing spaces.
         */
        TrimAll = TrimLeading | TrimTrailing
        };

        /**
           Trims leading and trailing whitespace from the input string
           and returns the number of whitespace characters removed.
         */
        std::size_t trim_string( std::string &, TrimPolicy = TrimAll );
        /**
           Trims leading and trailing whitespace from the input string
           and returns the trimmed string.
         */
        std::string trim_string( const std::string &, TrimPolicy = TrimAll );


        /**
           Attempts to remove all backslash-escaped chars from str.

           Removes backslash-escaped newlines from the input string, including
           any whitespace immediately following each backslash.

           The optional slash parameter defines the escape character.
        */
        std::size_t strip_slashes( std::string &str, const char slash = '\\' );

        /**
           Adds an escape sequence in front of any characters in
           instring which are also in the list of chars_to_escape.
           Returns the number of escapes added.

           e.g., to escape (with a single backslash) all $, % and \ in
           mystring with a backslash:

           <pre>
           escape_string( mystring, "$%\\", "\\" );
           </pre>

           (WARNING: the doxygen-generated HTML version of these docs
           may incorrectly show single backslashes in the above example!)


        */
        std::size_t escape_string( std::string & instring, const std::string & chars_to_escape, const std::string & escape_seq = "\\" );

        /**
           normalize_string() is like trim_string() and
           strip_slashes(), combined, plus it removes leading/trailing
           quotes:

           <pre>
           "this is a \
           sample multi-line, backslash-escaped \
           string."
           </pre>
           
           Will translate to:
           <pre>
           this is a sample multi-line, backslash-escaped string.
           </pre>
        */
        void normalize_string( std::string & );


        /**
           Returns the first whitespace-delimited token from the given
           string, or an empty string if there is no such token.
        */
        std::string first_token( const std::string & );

        /**
           Returns the passed-in string, minus the first
           whitespace-delimited token. An empty string is returned if
           there is no second token.
         */
        std::string after_first_token( const std::string & );



        /**
           Returns int values for chars '0'-'9', 'a'-'f' and 'A'-'F',
           else -1.
        */
        int int4hexchar( char character );

        /**
           Returns decimal value of wd, which is assumed to be a
           hex-encoded number. wd may optionally be prefixed with '#',
           as in \#ff00ff. Case is insignificant.

           On error -1 is returned, but -1 is also potentially a valid
           number, so there is really no way of knowing if it fails or
           not. :/
        */
        int hex2int( const std::string & wd );


        /**
           Lexically casts v to a string.
        */
        template <typename ValueT>
        std::string to( const ValueT & v )
        {
		return STPrivate::to_string(v);
        }

        /**
           Lexically casts v to a ValueT, or returns dflt if
           conversion fails.
        */
        template <typename ValueT>
        ValueT from( const std::string & v, const ValueT & dflt = ValueT() )
        {
                return STPrivate::from_string( v, dflt );
        }


        /**
           See translate_entities() for details.
        */
        typedef std::map<std::string,std::string> entity_map;


	/**
	   YAGNI!

	   A functor for translating entities in a set of strings.
	   Designed for use with std::for_each().
	*/
        struct entity_translator
        {
		/**
		   Sets the map and reverse options to be used from
		   calls to operator().
		*/
                entity_translator( const entity_map & map, bool reverse )
			: m_map(&map),m_rev(reverse)
                {
                }

		/**
		   Calls translate_entities( str, MAP, REVERSE ),
		   where MAP and REVERSE are the flags set via the
		   ctor.
		*/
                inline void operator()( std::string & str ) const
                {
                        translate_entities( str, *(this->m_map), this->m_rev );
                }
        private:
                const entity_map * m_map;
                bool m_rev;
                         
        };

        /**
           Internal-use initializer for setting up an entity
           translation map for default quote-escaping behaviour.
        */
        struct default_escapes_initializer
        {
		/**
		   Adds the following escape sequences to map:

		   - 1x backslash (\) == 2x backslash.

		   - 1x apostrophe  == 1x backslash 1x apostrophe

		   - 1x double-quote  == 1x backslash 1x double-quote  
		*/
                void operator()( entity_map & map );
        };


        /** Internal marker type. */
        template <typename ContextT> struct strtool_sharing_context {};

        /**
           Returns the default entity translation map, which can be used to
           [un]slash-escape the folling entities: '\\', '\'', '"'.
        */
        const entity_map & default_escapes_translations();

        /**
           Converts v to a string, applies translate_entities(...,trans,reverse ),
           and returns the resulting string.
        */
        template <typename ValueT>
        std::string translate( const ValueT & v,
                               const entity_map & trans,
                               bool reverse )
        {
                std::string val = to( v );
                translate_entities( val, trans, reverse );
                return val;
        }


        /**
           Calls translate( v,trans, false);
        */
        template <typename ValueT>
        std::string escape( const ValueT & v, const entity_map & trans = default_escapes_translations() )
        {
                return translate( v, trans, false );
        }


        /**
           Calls translate( v, trans, true );
        */
        template <typename ValueT>
        std::string unescape( const ValueT & v, const entity_map & trans = default_escapes_translations() )
        {
                return translate( v, trans, true );
        }

        /**
           Returns v as a quoted string, using the given quote
           character.
        */
        template <typename ValueT>
        std::string quote( const ValueT & v, const std::string & quote = "\'" )
        {
                return quote + to( v ) + quote;
        }

        /**
           Exactly like expand_dollar_refs_inline() but returns a new string
           which results from the expansions. The returned string may
           be the same as the original.
 
         */
        std::string expand_dollar_refs( const std::string & text, const entity_map & src );

        /**
           Parsed env vars out of buffer, replacing them with their
           values, as defined in the src map. Accepts variables
           in the format ${VAR} and $VAR.

           e.g., ${foo} corresponds to the value set in src["foo"].

           Referencing a variable which is not set does not
           expand the variable to an empty value: it is left
           as-is. Thus expanding ${FOO} when "FOO" is not set
           will result in "${FOO}".

           To get a dollar sign into the resulting string, escape
           it with a single backslash: this keeps it from being
           parsed as a ${variable}.

	   Returns the number of variables expanded.

	   Note that this function is much *more* efficient than using
	   translate_entities() to perform a similar operation.
	   Because of it's stricter format we can do a single pass
	   through the string and may not even have to reference the
	   source map.

	   Complexity depends on the number of ${vars} parts are expanded
	   in buffer: overall runtime depends on buffer length,
	   plus a non-determinate amount of time per ${var} expanded.

	   Design note: this really should be a function template,
	   accepting any lexically-castable key/val types, but the
	   function is quite long, and therefore not really suitable
	   to inclusion in the header.


	   Known misgivings:

	   - When buffer contains dollar signs which are preceeded by
	   a slash, the slash is stripped even if the $ does not
	   expand to anything. This is arguably behaviour.
        */
        std::size_t expand_dollar_refs_inline( std::string & buffer, const entity_map & src );


} } } // namespaces


#endif // s11n_net_s11n_STRINGTOOL_HPP_INCLUDED
