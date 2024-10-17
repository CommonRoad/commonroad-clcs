#ifndef s11n_net_s11n_VARIANT_HPP_INCLUDED
#define s11n_net_s11n_VARIANT_HPP_INCLUDED 1
////////////////////////////////////////////////////////////////////////
// variant.hpp: the s11n::variant class
////////////////////////////////////////////////////////////////////////


#include <string>
#include <sstream>
#include <map>

/**
   This file houses a little class for lexically casting strings and
   other types to and from each other.


   This software is released into the Public Domain by it's author,
   stephan beal (stephan@s11n.net).


Change history:

27 June 2005:
- Moved operator==() back into class, because they often cause
  odd overload ambiguities in unrelated code when comparing
  iterators of arbitrary types.

Early 2005:
- Switch to Marc Duerner's hack for the istream>> operator,
  which is the sanest one i can think of.

31 Dec 2004:
- Changed the impl of istream>>() again. Now uses a horrible
  hack but works across many more use cases.

28 Dec 2004:
- Changed the impl of istream>>() to use getc() instead of getline(),
  so newlines are preserved.


25 Nov 2004:
- Minor doc updates.
- Changed multiple-include guard to allow inclusion of this file twice
  for purposes of registering variant with s11n. Before, this header
  must have been included AFTER including s11n to pick up the
  registration. Now including this header after including s11n is safe
  if it has previously been included (and thus didn't pick up s11n
  registration).


2 Oct 2004:

- Accomodated recent changes in libs11n.


22 Aug 2004:

- Added ambiguity-buster overloads for operator==() for (const char *)
  and std::string.


20 Aug 2004:

- Added variant::empty()
- Moved variant::operator==() to a free function.
- Added variant::operator=(const char *) (see API notes for why).


17 Aug 2004:

- Initial release.
- After-relase:
- Added more docs.
- Added std::string and (const char *) overloads, to avoid some ambiguities.



16 Aug 2004:
- Zen Coding Session.

*/
namespace {} /* for doxygen */


namespace s11n { namespace Detail {

        namespace Private // not for use by clients of variant
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
			os.precision( 16 ); // unfortunate, but for the general case very useful
                        os << obj;
                        return os.str();
                }

// Why the hell doesn't overload selection take this one?
//                  inline std::string to_string( double d ) throw()
//                  {
//                          std::ostringstream os;
// 			 os.precision( 20 );
//                          os << d;
//                          return os.str();
//                  }


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


        } // end Private namespace



        /**
           variant provides a really convenient way to lexically cast
           strings and other streamable types to/from each other.

           All parameterized types used by this type must be:

           - i/o streamable. The operators must complement each other.

           - Assignable.

           - Default Constructable.

           This type is fairly light-weight, with only one std::string
           data member, so it should copy quickly and implicitely use
           std::string's CoW and reference counting features. Adding
           reference counting to this class would be of no benefit,
           and would probably hurt performance, considering that
           std::string's are optimized in these ways, and this type
           is simply a proxy for a std::string.

           For some uses the variant type can replace the requirement
           for returning a proxy type from a type's operator[](), as
           discussed in Scott Meyers' <em>More Effective C++</em>, Item
           30. This class originally was such a proxy, and then
           evolved into a generic solution for POD-based types, which
           inherently also covers most i/ostreamable types. It is less
           efficient than specialized proxies for, e.g. (char &), but
           it is also extremely easy to use, as shown here:


<pre>
        s11n::variant lex = 17;
        int bogo = lex;
        ulong bogol = bogo * static_cast<long>(lex);
        lex = "bogus string";

        typedef std::map<variant,variant> LMap;
        LMap map;
        
        map[4] = "one";
        map["one"] = 4;
        map[123] = "eat this";
        map['x'] = "marks the spot";
        map["fred"] = 94.3 * static_cast<double>( map["one"] );
        map["fred"] = 10 * static_cast<double>( map["fred"] );
        map["123"] = "this was re-set";
        int myint = map["one"];

</pre>

        Finally, Perl-ish type flexibility in C++. :)

        It gets better: if we're using s11n, we can now save and load
        these objects at will:

<pre>
        s11nlite::save( map, "somefile.s11n" );
        ...

        LMap * map = s11nlite::load_serializable<LMap>( "somefile.s11n" );
</pre>

        */
        struct variant
        {
                /**
                   Constructs an empty object. Calling <code>cast_to<T>()</code>
                   on an un-populated variant object will return T().
                */
                variant() throw(){}

                ~variant() throw() {}

                /**
                   Lexically casts f to a string.
                */
                template <typename FromT>
                variant( const FromT & f ) throw()
                {
                        this->m_data = Private::to_string( f );
                }

                /**
                   Efficiency overload.
                */
                variant( const std::string & f ) throw() : m_data(f)
                {
                }
                /**
                   See operator=(const char *) for a note about why
                    this exists.
                 */
                variant( const char * str ) throw() : m_data(str?str:"")
                {
                }


                /**
                   Copies rhs's data to this object.
                */
                variant( const variant & rhs ) throw() : m_data(rhs.m_data)
                {
                }

                /**
                 Returns (this-&lt;str() &lt; rhs.str()).
                */
                inline bool operator<( const variant & rhs ) const
                {
                        return this->str() < rhs.str();
                }

                /**
                 Returns (this-&lt;str() > rhs.str()).
                */
                inline bool operator>( const variant & rhs ) const
                {
                        return this->str() > rhs.str();
                }


                /**
                   Copies rhs's data and returns this object.
                */
                inline variant & operator=( const variant & rhs ) throw()
                {
                        if( &rhs != this ) this->m_data = rhs.m_data;
                        return *this;
                }

                /**
                   This overload exists to keep the compiler/linker
                   from generating a new instantiation of this function
                   for each differently-lengthed (const char *)
                   which is assigned to a variant.
                */
                inline variant & operator=( const char * rhs ) throw()
                {
                        this->m_data = rhs ? rhs : "";
                        return *this;
                }


                /**
                   lexically casts str() to a ToType, returning
                   dflt if the cast fails.

                   When calling this function you may need to use the
                   following syntax to avoid compile errors:

                   Foo foo = lex.template cast_to<Foo>();

                   (It's weird, i know, and the first time i saw it,
                   finding the solution to took me days. (Thank you,
                   Nicolai Josuttis!))

                   However, in normal usage you won't need to use this
                   function, as the generic type conversion operator
                   does the exact same thing:

<pre>
variant lex = 17;
int foo = lex;
</pre>
                */
                template <typename ToType>
                ToType cast_to( const ToType & dflt = ToType() ) const throw()
                {
                        return Private::from_string( this->m_data, dflt );
                }


                /**
                   i used to LOVE C++... After writing this function i
                   WORSHIP C++. The grace with which C++ handles this
                   is pure magic, my friends.

                   16.8.2004 ----- stephan
                */
                template <typename ToType>
                inline operator ToType() const throw()
                {
                        return this->template cast_to<ToType>();
                }

                /**
                   Overload to avoid ambiguity in some cases.
                */
                inline operator std::string () const throw()
                {
                        return this->str();
                }
                /**
                   Returns the same as str().
                */
                operator std::string & () throw() { return  this->m_data; }

                /**
                   Returns the same as str().
                */
                operator const std::string & () const throw() { return  this->m_data; }


                /**
                   Overload to avoid ambiguity in some cases.
                   Useful for mixing C and C++ APIs:

<pre>
variant arg = "USER";
variant user = ::getenv(arg);
</pre>
                */
                inline operator const char * () const throw()
                {
                        return this->str().c_str();
                }



                /**
                   Sets this object's value and returns this object.
                */
                template <typename ToType>
                inline variant & operator=( const ToType & f ) throw()
                {
                        this->m_data = Private::to_string( f );
                        return *this;
                }

                /**
                   Returns a reference to this object's raw string
                   data.
                */
                inline std::string & str() throw() { return  this->m_data; }


                /**
                   Returns a const reference to this object's raw
                   string data.
                */
                inline const std::string & str() const throw() { return  this->m_data; }

                /**
                   Returns true if this object contains no data, else false.
                 */
                inline bool empty() const { return this->m_data.empty(); }

		/**
		   Casts lhs to a T object and returns true only if that
		   object compares as equal to rhs.
		*/
		template <typename T>
		inline bool operator==( const T & rhs ) const
		{
			return this->template cast_to<T>() == rhs;
		}
		template <typename T>
		inline bool operator!=( const T & rhs ) const
		{
		    return !(this->operator==( rhs ));
		}

		/** Returns lhs.str() == rhs.str(). */
		inline bool operator==( const variant & rhs ) const
		{
			return this->str() == rhs.str();
		}
		inline bool operator!=( const variant & rhs ) const
		{
		    return !(this->str() == rhs.str());
		}

		/**
		   Avoid an ambiguity...
		   
		   If rhs == 0 then this function returns true if
		   lhs.empty().
		*/
		inline bool operator==( const char * rhs ) const
		{
			if( ! rhs ) return this->empty();
			return this->str() == std::string(rhs);
		}

		/**
		   Avoid an ambiguity...
		*/
		inline bool operator==( const std::string & rhs ) const
		{
			return this->str() == rhs;
		}


		/**
		   Copies this->str() to os.
		*/
		inline std::ostream & operator<<( std::ostream & os ) const
		{
			return os << this->str();
		}

		/**
		   Reads from the input stream, appending to this->str() until the stream
		   gives up. If the implementation of this function seems "wrong"
		   to you, please read the justification in this paper:

		   http://s11n.net/papers/lexical_casting.html

		   i believe there to be no satisfying, truly generic
		   implementation for this function :(.
		*/
		inline std::istream & operator>>( std::istream & is )
		{
			/****
			     WTF doesn't this do anything?!?!?!?!?
			     
			     int c;
			     while( is.get(c).good() ) { 
			     a.str() += c;
			     }
			****/
			/*****
			      while( std::getline( is, a.str() ).good() );
			      // ^^^ eeek! strips newlines!
			      *****/
			/*****
			      is >> a.str(); // depends on skipws.
			*****/
			/*****
			      std::getline( is, a.str(), '\v' ); // UGLY, EVIL hack!
			      The \v char ("virtical tab") is an ugly hack: it is simply a char from
			      the ascii chart which never shows up in text. At least, i hope it
			      doesn't. AFIK, \v was historically used on old line printers and some
			      ancient terminals, but i've never seen it actually used. Unicode maps
			      0-255 to the same as ascii, so this shouldn't be a problem for
			      unicode.
			*****/
			// Many thanks to Marc Duerner:
			std::getline( is, this->str(), static_cast<std::istream::char_type>(std::istream::traits_type::eof()) );
			return is;
		}
		

        private:
                std::string m_data;
        };





} } // namespace s11n::Detail


#endif // s11n_net_s11n_VARIANT_HPP_INCLUDED

