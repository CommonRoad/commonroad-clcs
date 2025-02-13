#ifndef s11n_net_s11n_v1_1_EXCEPTION_HPP_INCLUDED
#define s11n_net_s11n_v1_1_EXCEPTION_HPP_INCLUDED 1

#include <string>
#include <exception>
#include <s11n.net/s11n/s11n_config.hpp>

//Added by Damien to make Windows compile work
#include <s11n.net/s11n/export.hpp> // S11N_EXPORT_API

namespace s11n {

	/**
	   The base-most exception type used by s11n.
	*/
   //Added by Damien to make Windows compile work 
        struct S11N_EXPORT_API s11n_exception : public std::exception
        {
	public:
		virtual ~s11n_exception() throw() {}
		// removed in 1.2.6: explicit s11n_exception( const std::string & What );

		/**
		   Creates an exception with the given formatted
		   what() string.  Takes a printf-like format
		   string. If the expanded string is longer than some
		   arbitrarily-chosen internal limit [hint: 2k bytes]
		   then it is truncated.

 		   If you get overload ambiguities with the
 		   std::string-argument ctor, this is because you've
 		   passed a (char const *) string to those ctors and
 		   relied on implicit conversion to std::string.
 		   Simply wrapping those c-strings in std::string
 		   ctors should get around the problem.

		   Historical note:

		   This ctor, introduced in version 1.2.6, conflicted
		   with an older 3-arg ctor taking (std::string,char
		   const *,uint) arguments, but this one is far more
		   flexible, so the older was removed. We also had
		   ctor taking a std::string, but that was removed
		   to avoid additional ambiguities.
		*/
   		explicit s11n_exception( const char *format, ... );

		/**
		   Returns the 'what' string passed to the ctor.
		*/
                virtual const char * what() const throw();
	protected:
		/**
		   Intended to be used by ctors.
		*/
		void what( std::string const & ) throw();
		s11n_exception();
        private:
                std::string m_what;
        };

	/**
	   An exception type for classloader-related exceptions. These
	   need to be caught separately from s11n_exceptions in some
	   cases because sometimes a classloader can try other
	   alternatives on an error.
	*/
	struct S11N_EXPORT_API factory_exception : public s11n_exception
	{
	public:
		virtual ~factory_exception() throw() {}
		// removed in 1.2.6: explicit factory_exception( const std::string & What ) : s11n_exception( What ) {}
   		explicit factory_exception( const char *format, ... );
//                 factory_exception( const std::string & What, const std::string & file, unsigned int line ) : s11n_exception( What,file,line ) {}
	};


	/**
	   Really for use by clients, i/o layers, and s11nlite, not by
	   the s11n core.
	*/
	struct io_exception : public s11n_exception
	{
	public:
		virtual ~io_exception() throw() {}
		// removed in 1.2.6: explicit io_exception( const std::string & What ) : s11n_exception( What ) {}
   		explicit io_exception( const char *format, ... );
	};


} // namespace s11n

// /**
//    S11N_THROW(WHAT) simply throws s11n_exception(WHAT,__FILE__,__LINE__).
// */
// #define S11N_THROW(WHAT) throw ::s11n::s11n_exception(std::string(WHAT),__FILE__,__LINE__)

#endif // s11n_net_s11n_v1_1_EXCEPTION_HPP_INCLUDED
