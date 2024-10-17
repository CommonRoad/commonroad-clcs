#ifndef s11n_net_s11n_v1_1_ALGO_TPP_INCLUDED
#define s11n_net_s11n_v1_1_ALGO_TPP_INCLUDED
// implementation code for non-inlined code declared in algo.hpp
#include <s11n.net/s11n/variant.hpp>


template <typename DataNodeType, typename SerializableT >
bool s11n::serialize_subnode( DataNodeType & target,
			      const std::string & nodename,
			      const SerializableT & src )
{
	typedef ::s11n::node_traits<DataNodeType> NT;
	std::unique_ptr<DataNodeType> sub(NT::create(nodename));
	typedef typename ::s11n::type_traits<SerializableT>::type ST;
	if( serialize<DataNodeType,ST>( *sub, src ) )
	{
		NT::children(target).push_back( sub.release() ); // take over ownership
		return true;
	}
	return false;
}

template <typename DataNodeType, typename DeserializableT>
bool s11n::deserialize_subnode( const DataNodeType & src,
				const std::string & subnodename,
				DeserializableT & target )
{
	const DataNodeType * ch = find_child_by_name( src, subnodename );
	if( ! ch ) return false;
	typedef typename ::s11n::type_traits<DeserializableT>::type ST;
	return deserialize<DataNodeType,ST>( *ch, target );
}

template <typename DataNodeType, typename DeserializableT>
DeserializableT *
s11n::deserialize_subnode( const DataNodeType & src,
			   const std::string & subnodename )
{
	const DataNodeType * ch = ::s11n::find_child_by_name( src, subnodename );
	if( ! ch ) return false;
	return deserialize<DataNodeType,DeserializableT>( *ch );
}

template <typename NodeT>
NodeT *
s11n::find_child_by_name( NodeT & parent, const std::string & name )
{
	typedef node_traits<NodeT> TR;
	typedef typename NodeT::child_list_type::iterator IT;
	IT it = std::find_if( TR::children(parent).begin(),
			      TR::children(parent).end(),
			      ::s11n::Detail::same_name<NodeT>( name )
			      );
	return (TR::children(parent).end() == it) ? 0 : *it;
}

template <typename NodeT>
const NodeT *
s11n::find_child_by_name( const NodeT & parent, const std::string & name )
{
	typedef node_traits<NodeT> TR;
	typedef typename NodeT::child_list_type::const_iterator CIT;
	CIT it = std::find_if( TR::children(parent).begin(),
                                       TR::children(parent).end(),
			       ::s11n::Detail::same_name<NodeT>( name )
			       );
	return (TR::children(parent).end() == it) ? 0 : *it;
}

        template <typename NodeType>
        NodeType & s11n::create_child( NodeType & parent,
				 const std::string nodename )
        {
                typedef ::s11n::node_traits<NodeType> NTR;
                NodeType * n = NTR::create( nodename );
                NTR::children( parent ).push_back( n );
                return *n;
        }


        template <typename NodeT, typename DestContainerT>
        size_t s11n::find_children_by_name( const NodeT & parent,
				      const std::string & name,
				      DestContainerT & target )
        {
		typedef node_traits<NodeT> TR;
                size_t c = target.size();
                ::s11n::copy_if( TR::children(parent).begin(),
				 TR::children(parent).end(),
#if defined (__QNXNTO__)
				 // Weird QNX workaround submitted by James Ingraham. Added in 1.2.5:
				 inserter( target, target.begin() ),
#else
				 std::insert_iterator<DestContainerT>( target,
								       target.begin() ),
#endif
				 ::s11n::Detail::same_name<NodeT>( name )
                                         );
                // you gotta love the STL, man.
                return target.size() - c;
        }


template <typename NodeT>
void s11n::debug::dump_node_structure( const NodeT & n, int indentlv)
{
	typedef s11n::node_traits<NodeT> TR;
	typedef typename TR::child_list_type CLT;
	typedef typename CLT::const_iterator CIT;
	CIT b = TR::children(n).begin();
	CIT e = TR::children(n).end();
	std::cerr << "Node: name="<<TR::name(n)<<", class="<<TR::class_name(n) << "\n";
	++indentlv;
			for( ; e != b; b++ )
			{
				for( int i = 0; i < indentlv; i++ )
				{
					std::cerr << "    ";
				}
				std::cerr << '+';
				dump_node_structure( *(*b), indentlv );
			}
}

template <typename NodeType, typename VersionType, typename SerType>
bool s11n::serialize_versioned( NodeType & dest, VersionType const ver, SerType const & src )
{
    typedef node_traits<NodeType> NTR;
    NTR::set( dest, "version", ver );
    NodeType & ch( create_child( dest, "vdata" ) );
    return serialize( ch, src );
}

template <typename NodeType, typename VersionType, typename SerType>
bool s11n::deserialize_versioned( NodeType const & src, VersionType const ver, SerType & dest )
{
    typedef node_traits<NodeType> NTR;
    VersionType gotver( NTR::get( src, "version", VersionType() ) );
    if( ver !=  gotver )
    {
	using s11n::Detail::variant;
	variant v1( ver );
	variant v2( gotver );
	throw s11n_exception( "Version mismatch. Expected '%s' but got '%s'.",
			      v1.str().c_str(),
			      v2.str().c_str() );
    }
    NodeType const * ch = find_child_by_name( src, "vdata" );
    if( ! ch )
    {
	throw s11n_exception( "Data version matched but versioned data not found." );
    }
    return deserialize( *ch, dest );
}

#endif // s11n_net_s11n_v1_1_ALGO_TPP_INCLUDED
