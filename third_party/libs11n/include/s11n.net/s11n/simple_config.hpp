#ifndef s11n_SIMPLECONFIG_HPP_INCLUDED
#define s11n_SIMPLECONFIG_HPP_INCLUDED 1
// Utility class for use with s11nlite.


#include <s11n.net/s11n/io/strtool.hpp>
#include <s11n.net/s11n/s11n_debuggering_macros.hpp> // CERR
#include <s11n.net/s11n/exception.hpp>
#include <stdlib.h> // getenv()
#include <stdexcept>
#include <sstream>
namespace s11nlite {

	/**
	   simple_config provides very basic config file features.
	*/
	class simple_config
	{
	public:
		typedef ::s11nlite::node_type node_type;
		typedef ::s11nlite::node_traits node_traits;
	private:
		std::string m_base;
		std::string m_abs;
		node_type * m_node;
		/** Tried to load this object's file. */
		void load_node()
		{
			try
			{
				this->m_node = ::s11nlite::load_node( this->m_abs );
			}
			catch(...)
			{
				this->m_node = 0;
			}
			if( ! this->m_node )
			{
				this->m_node = node_traits::create( this->m_base );
				if( 0 ) CERR << "s11nlite::simple_config error: could not load node from file "
				     << "'" << this->m_abs << "'.\n"
				     << "Creating new node named '"<< this->m_base << "'."
				     << "\n";
			}
		}

		/** Resolves basename() to an absolute path under $HOME, or throws on error. */
		void resolve_path() throw(std::runtime_error)
		{
			if( this->m_base.empty() ||
			    (std::string::npos != this->m_base.find( '/' )) )
			{
				throw std::runtime_error( "s11n::util::simple_config: invalid base filename: "+this->m_base );
			}
			const char * home = ::getenv("HOME");
			if( ! home )
			{
				throw std::runtime_error( "s11n::util::simple_config: $HOME is not set! Cannot create config file!" );
			}
			::s11n::io::strtool::entity_map env;
			env["HOME"] = home;
			std::string tmp = "${HOME}/."+m_base+".s11n";
			this->m_abs = ::s11n::io::strtool::expand_dollar_refs( tmp, env );
			//CERR << "abs filename="<<this->m_abs << "\n";
		}
	public:
		/**
		   Constructs a config object associated with the file
		   $HOME/.basename.s11n. If the file does not exist we
		   assume we can create it. If $HOME cannot be
		   resolved or basename has a '/' character in it an
		   exception is thrown. basename should be a filename
		   friendly string, such as "myapp", "My_App", or
		   "My_App-1.0.3".
		*/
		simple_config( const std::string & basename ) throw(std::runtime_error)
			: m_base(basename),m_abs(),m_node(0)
		{
			this->resolve_path();
			this->load_node();
		}

		/**
		   Attempts to save this object's node() to this->abs_path().
		*/
		~simple_config() throw()
		{
			if( ! this->m_abs.empty() )
			{
				try
				{
					s11nlite::save( *this->m_node, this->m_abs );
				}
				catch(...)
				{
					CERR << "EXCEPTION while saving simple_config to " << this->m_abs;
				}
			}
			delete this->m_node;
		}


		/**
		   Absolute path to this object's file.
		*/
		std::string abs_path() const
		{
			return this->m_abs;
		}

		/**
		   Base filename passed to the ctor.
		*/
		std::string basename() const
		{
			return this->m_base;
		}

		/**
		   This object's data node, which can be used to
		   store any s11nlite-serializable data.
		*/
		node_type & node()
		{
			return *this->m_node;
		}
	};

} // namespace


#endif // s11n_SIMPLECONFIG_HPP_INCLUDED
