////////////////////////////////////////////////////////////////////////
// platform-specific parts of plugin.{c,h}pp for unknown platforms.
////////////////////////////////////////////////////////////////////////

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
		//std::string where = find( basename );
		m_dll_error = std::string("s11n::plugin::open(")
			+ basename
			+ std::string( "): not implemented on this platform." );
		return std::string();
	}

}} // namespace

