#include <s11n.net/s11n/exception.hpp>
#include <s11n.net/s11n/s11n_config.hpp>
#include <sstream>

#include <stdio.h> // vsnprintf()

// #include <s11n.net/s11n/s11n_debuggering_macros.hpp> // CERR
// #include <iostream>

#include <stdarg.h> // va_list

/**
s11n_CONFIG_EXCEPTION_BUFFER_SIZE sets the maximum size of expanded format strings
for exception() and set_pending_exception().
*/
#define s11n_CONFIG_EXCEPTION_BUFFER_SIZE (1024*4)

namespace s11n {


	s11n_exception::s11n_exception() : m_what()
	{
	}

	void s11n_exception::what( std::string const & w) throw()
	{
		this->m_what = w;
	}

	const char * s11n_exception::what() const throw()
	{
		return this->m_what.empty()
			? 0
			: this->m_what.c_str();
	}

	/**
	   Internal helper to avoid code duplication.

	   buffsize is the maximum size of expanded format strings
	   for vsnprintf().

	   format is the format string for vnsprintf().

	   vargs are the args for vsnprintf().

	   Usage:

	   \code
	va_list vargs;
	va_start( vargs, format );
	mystring = local_sprintf( 1024 * 2, format, vargs);
	va_end(vargs);
	\endcode

	Returns an empty string on error.

	If the expanded string is larger than buffsize then the last 3
	characters of the string are replaced with '...' to show that
	the text continues.
	*/
	static std::string local_sprintf( const int buffsize,
					  const char *format,
					  va_list vargs ) throw()
	{
		// Pedantic note: this code was taken from SpiderApe,
		// which is released under the Mozilla Public
		// License/LPGL. However, i wrote that code, so i may
		// re-use it here (in the Public Domain) however i
		// damned well please. :)
		if( (! format) || (buffsize<=0) )
		{
			return std::string();
		}
		char buffer[buffsize];
		int size = vsnprintf(buffer, buffsize, format, vargs);
		if( size <= 0 )
		{
			return std::string();
		}
		if (size > (buffsize-1))
		{
			// replace tail of msg with "..."
			size = buffsize-1;
			int i = buffsize-4;
			if( i < 0 ) i = 0;
			for( ; i < buffsize-1; ++i )
			{
				buffer[i] = '.';
			}
		}
		buffer[size] = '\0';
		return std::string(buffer);
	}


	s11n_exception::s11n_exception( const char *format, ... )
		: m_what()
	{
		va_list vargs;
		va_start( vargs, format );
		this->what( local_sprintf(s11n_CONFIG_EXCEPTION_BUFFER_SIZE,
					  format, vargs) );
		va_end(vargs);
	}

	factory_exception::factory_exception( const char *format, ... )
	{
		va_list vargs;
		va_start( vargs, format );
		this->what( local_sprintf(s11n_CONFIG_EXCEPTION_BUFFER_SIZE,
					  format, vargs) );
		va_end(vargs);
	}
	io_exception::io_exception( const char *format, ... )
	{
		va_list vargs;
		va_start( vargs, format );
		this->what( local_sprintf(s11n_CONFIG_EXCEPTION_BUFFER_SIZE,
					  format, vargs) );
		va_end(vargs);
	}

#undef s11n_S11N_EXCEPTION_VARARGS_CTOR_IMPL

} // namespace s11n
