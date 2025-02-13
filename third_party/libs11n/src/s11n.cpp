
#include <s11n.net/s11n/s11n.hpp>
#include <s11n.net/s11n/s11n_config.hpp> // s11n_PACKAGE_VERSION
#include <s11n.net/s11n/s11n_debuggering_macros.hpp>

namespace s11n {

	namespace Private
	{
		void s11n_1_1_assertion()
		{
		}
	}

        std::string library_version()
        {
                return s11n_S11N_LIBRARY_VERSION;
        }

	namespace debug
	{
		static unsigned long _trace_mask = S11N_TRACE_LEVELS;
		unsigned long trace_mask( unsigned long f )
		{
			unsigned long old = _trace_mask;
			_trace_mask = f;
			return old;
		}
		unsigned long trace_mask()
		{
			return _trace_mask;
		}


		static std::ostream * m_trace_stream = 0;
		std::ostream & trace_stream()
		{
			return m_trace_stream ? *m_trace_stream : std::cerr;
		}

		void trace_stream( std::ostream & os )
		{
			m_trace_stream = &os;
		}

		trace_mask_changer::trace_mask_changer()
		{
			this->m_mask = trace_mask();
		}
		trace_mask_changer::trace_mask_changer( unsigned long m )
		{
			this->m_mask = trace_mask();
			trace_mask( m );
		}
		trace_mask_changer::~trace_mask_changer()
		{
			trace_mask( this->m_mask );
		}

	}

} // namespace s11n
