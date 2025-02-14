
#include <algorithm> // for_each
#include <iterator>

#include <s11n.net/s11n/algo.hpp>
#include <s11n.net/s11n/s11n_node.hpp>

#include <s11n.net/s11n/s11n_debuggering_macros.hpp>

namespace s11n {
#define NODE_CLASS_NAME "s11n::s11n_node"
// note that the default implclass is not useable for purposes of classloading
// serializables, but:
//  a) the parsers need some text to read.
//  b) "raw" nodes are often useful for de/ser'ing maps, lists, etc.
//  c) Serializables must set their implclass *anyway*, so they will
//     (or should) always overwrite the default class.

	using namespace ::s11n::debug;

	s11n_node::s11n_node( const std::string & name ) : m_name(name),m_iname(NODE_CLASS_NAME)
	{
		S11N_TRACE(TRACE_CTOR) << "creating s11n_node("<<name<<") @ " << std::hex << this << '\n';
	}
	s11n_node::s11n_node( const std::string & name, const std::string implclass ) : m_name( name ), m_iname( implclass )
	{
		S11N_TRACE(TRACE_CTOR) << "creating s11n_node("<<name<<","<<implclass<<") @ " << std::hex << this << '\n';
	}

	s11n_node::s11n_node() : m_name("s11n_node"), m_iname(NODE_CLASS_NAME)
	{
		S11N_TRACE(TRACE_CTOR) << "creating s11n_node() @ " << std::hex << this << '\n';
	}

        s11n_node & s11n_node::operator=( const s11n_node & rhs )
        {
                if( &rhs == this ) return *this;
                this->copy( rhs );
                return *this;
        }
        s11n_node::s11n_node( const s11n_node & rhs )
        {
                if( &rhs == this ) return;
                this->copy( rhs );
        }
	s11n_node::~s11n_node()
	{
		S11N_TRACE(TRACE_DTOR) << "~s11n_node @ " << std::hex << this << '\n';
                this->clear_children();
	}

        s11n_node::map_type & s11n_node::properties()
        {
                return this->m_map;
        }

        const s11n_node::map_type & s11n_node::properties() const
        {
                return this->m_map;
        }


	void s11n_node::swap( s11n_node & rhs )
	{
		this->children().swap( rhs.children() );
		this->properties().swap( rhs.properties() );
		this->m_name.swap( rhs.m_name );
		this->m_iname.swap( rhs.m_iname );
	}


        void s11n_node::copy( const s11n_node & rhs )
        {
		if ( &rhs == this ) return;
                this->clear();
                this->name( rhs.name() );
                this->class_name( rhs.class_name() );
                std::copy( rhs.properties().begin(), rhs.properties().end(),
                           std::insert_iterator<map_type>( this->m_map, this->m_map.begin() )
                           );
                std::for_each( rhs.children().begin(),
                               rhs.children().end(),
                               Detail::child_pointer_deep_copier<child_list_type>( this->children() )
                               );
        }

	s11n_node::child_list_type & s11n_node::children()
	{
                return this->m_children;
	}

	const s11n_node::child_list_type & s11n_node::children() const
	{
                return this->m_children;
	}

	void s11n_node::clear()
	{
                this->clear_children();
		this->clear_properties();
	}

	void s11n_node::class_name( const std::string & n )
	{
		this->m_iname = n;
	}

	std::string
        s11n_node::class_name()const
	{
		return this->m_iname;
	}


	bool s11n_node::empty() const
	{
		return this->children().empty()
			&& this->properties().empty();
	}

	void
        s11n_node::name( const std::string & n )
	{
		this->m_name = n;
	}

	std::string
        s11n_node::name() const
	{
		return this->m_name;
	}


	bool s11n_node::is_set( const std::string & key ) const
	{
                return this->m_map.end() != this->m_map.find( key );
	}

	void s11n_node::unset( const std::string & key )
	{
		s11n_node::map_type::iterator iter;
		iter = this->m_map.find( key );
		if ( iter == this->m_map.end() ) return;
		this->m_map.erase( iter );
		return;
	}

	void s11n_node::clear_properties()
	{
		if ( m_map.empty() ) return;
		m_map.erase( m_map.begin(), m_map.end() );
	}

	void s11n_node::clear_children()
	{
                std::for_each( this->children().begin(),
                               this->children().end(),
                               object_deleter() );
                this->children().clear();
        }



}  // namespace s11n

#undef NODE_CLASS_NAME
