#ifndef s11n_net_s11n_v1_1_MAP_TPP_INCLUDED
#define s11n_net_s11n_v1_1_MAP_TPP_INCLUDED
// This file holds the implementations for template code declared
// in mapish.hpp.

#include <s11n.net/s11n/variant.hpp> // lexical casting
#include <s11n.net/s11n/serialize.hpp> // core serialize funcs
#include <s11n.net/s11n/exception.hpp> // s11n_exception class
#include <s11n.net/s11n/abstract_creator.hpp> // abstract_creator class

template <typename NodeType, typename MapType>
bool s11n::map::serialize_streamable_map( NodeType & dest, const MapType & src )
{
	typedef ::s11n::node_traits<NodeType> TR;
	typedef typename MapType::value_type VT;
	typedef typename VT::first_type FT;
	typedef typename VT::second_type ST;
	VT p;
	typename MapType::const_iterator it = src.begin();
	::s11n::Detail::variant v1, v2;
	for( ; src.end() != it; ++it )
	{
		v1 = (*it).first;
		v2 = (*it).second;
		TR::set(dest, v1.str(), v2.str() );
	}
	return true;
}


template <typename NodeType, typename MapType>
bool s11n::map::serialize_streamable_map( NodeType & dest,
					  const std::string & subnodename,
					  const MapType & src )
{
	typedef ::s11n::node_traits<NodeType> TR;
	std::unique_ptr<NodeType> ch( TR::create( subnodename ) );
	if( serialize_streamable_map( *ch, src ) )
	{
		TR::children(dest).push_back( ch.release() );
	}
	return 0 == ch.get();
}

template <typename NodeType, typename MapType>
bool s11n::map::deserialize_streamable_map( const NodeType & src, MapType & dest )
{
	typedef ::s11n::node_traits<NodeType> NTR;
	typedef typename MapType::value_type VT; // pair
	typedef typename VT::first_type T1;
	typedef typename VT::second_type T2;

	const T1 default1 = T1();
	const T2 default2 = T2();
	typedef typename NTR::property_map_type PMT;
	typedef typename PMT::const_iterator CIT;
	CIT it = NTR::properties(src).begin();
	CIT et = NTR::properties(src).end();
	::s11n::Detail::variant v1, v2;
	for( ; et != it; ++it )
	{
		v1 = (*it).first;
		v2 = (*it).second;
		dest.insert( std::make_pair( v1.template cast_to<T1>( default1 ),
					     v2.template cast_to<T2>( default2 )
					     ) );
	}
	return true;
}

template <typename NodeType, typename MapType>
bool s11n::map::deserialize_streamable_map( const NodeType & src,
					    const std::string & subnodename,
					    MapType & dest )
{
	const NodeType * ch = ::s11n::find_child_by_name( src, subnodename );
	if( ! ch ) return false;
	return deserialize_streamable_map<NodeType,MapType>( *ch, dest );
}
		




template <typename NodeType, typename PairType>
bool s11n::map::serialize_streamable_pair( NodeType & dest, const PairType & src )
{
	typedef s11n::node_traits<NodeType> TR;
	typedef ::s11n::s11n_traits<PairType> STR;
	TR::class_name( dest, STR::class_name(&src) );
	typedef typename PairType::first_type FT;
	typedef typename PairType::second_type ST;
	TR::set( dest, "first", src.first );
	TR::set( dest, "second", src.second );
	return true;
}

template <typename PairType, typename NodeType>
PairType s11n::map::deserialize_streamable_pair( const NodeType & src  )
{
	typedef ::s11n::node_traits<NodeType> TR;
	if( ! TR::is_set( src, "first" ) )
	{
		throw ::s11n::s11n_exception( "%s:%d: deserialize_streamable_pair(): 'first' property not set!", __FILE__, __LINE__ );
	}
	if( ! TR::is_set( src, "second" ) )
	{
		throw ::s11n::s11n_exception( "%s:%d: deserialize_streamable_pair(): 'second' property not set!", __FILE__, __LINE__ );
	}

	typedef typename PairType::first_type T1;
	typedef typename PairType::second_type T2;
	T1 default1 = T1();
	T2 default2 = T2();
	return std::make_pair( TR::get( src, "first", default1 ),
			       TR::get( src, "second", default2 )
			       );
}


template <typename NodeType, typename PairType>
bool s11n::map::deserialize_streamable_pair( const NodeType & src, PairType & dest )
{
	typedef ::s11n::node_traits<NodeType> TR;
	typedef typename PairType::first_type FT;
	typedef typename PairType::second_type ST;
	if( ! TR::is_set( src, "first" ) )
	{
		throw ::s11n::s11n_exception( "%s:%d: deserialize_streamable_pair(): 'first' property not set!", __FILE__, __LINE__ );
	}
	if( ! TR::is_set( src, "second" ) )
	{
		throw ::s11n::s11n_exception( "%s:%d: deserialize_streamable_pair(): 'second' property not set!", __FILE__, __LINE__ );
	}
	dest.first = TR::get( src, "first", dest.first );
	dest.second = TR::get( src, "second", dest.second );
	return true;
}

template <typename NodeType, typename MapType>
bool s11n::map::serialize_streamable_map_pairs( NodeType & dest, const MapType & src )
{
	typedef s11n::node_traits<NodeType> TR;
	typedef ::s11n::s11n_traits<MapType> STR;
	TR::class_name( dest, STR::class_name(&src) );
	typedef typename MapType::value_type VT;
	typedef typename VT::first_type FT;
	typedef typename VT::second_type ST;
	typename MapType::const_iterator it = src.begin();
	for( ; src.end() != it; ++it )
	{
		if( ! serialize_streamable_pair( create_child( dest, "pair" ), *it ) )
		{
			return false;
		}
	}
	return true;
}


template <typename NodeType, typename MapType>
bool s11n::map::deserialize_streamable_map_pairs( const NodeType & src, MapType & dest )
{
	typedef typename MapType::value_type VT; // pair
	typedef typename VT::first_type T1;
	typedef typename VT::second_type T2;
	typedef std::list<const NodeType *> ChList;
	typedef typename ChList::const_iterator ChIt;
			
	static const T1 default1 = T1();
	static const T2 default2 = T2();
			
	//::s11n::object_reference_wrapper<MapType> dwrap(dest); // in case dest is a pointer type
			
	ChList namedch;
	if( 0 == ::s11n::find_children_by_name( src, "pair", namedch ) ) return true; // fine
	ChIt it = namedch.begin();
	ChIt et = namedch.end();
	if( et == it ) return true;
	typedef ::s11n::node_traits<NodeType> TR;
	const NodeType * ch = 0;
	VT p = VT(default1,default2);
	MapType buf;
	try
	{
		for( ; et  != it; ++it )
		{
			ch = *it;
			buf.insert( deserialize_streamable_pair<VT>( *ch ) );
		}
	}
	catch(...)
	{
		using namespace ::s11n::debug;
		S11N_TRACE(TRACE_ERROR) << "deserialize_streamable_map_pairs(): srcnode="<<std::dec<<&src<<": propagating exception\n";
		throw;
	}
	dest.swap(buf);
	return true;
}


template <typename NodeType, typename PairType>
bool s11n::map::serialize_pair( NodeType & dest, const PairType & src )
{
	typedef s11n::node_traits<NodeType> NT;
	typedef ::s11n::s11n_traits<PairType> STR;
	typedef typename PairType::first_type FstT;
	typedef typename PairType::second_type SndT;
	NT::class_name( dest, STR::class_name(&src) );
	std::unique_ptr<NodeType> ch( NT::create("first") );
	//                         if( ! ::s11n::serialize<NodeType,FstT>( *ch, src.first ) )
	// Compiler bug??? ^^^^ fully qualifying that causes a compile error, but doing the
	// same thing on .second below does not!
	if( ! serialize( *ch, src.first ) )
	{
		return false;
	}
	std::unique_ptr<NodeType> ch2( NT::create("second") );
	if( ! ::s11n::serialize<NodeType,SndT>( *ch2, src.second ) )
	{
		return false;
	}
	NT::children(dest).push_back( ch.release() );
	NT::children(dest).push_back( ch2.release() );
	return true;
}


template <typename NodeType, typename PairType>
bool s11n::map::deserialize_pair( const NodeType & src, PairType & dest )
{
	typedef ::s11n::node_traits<NodeType> NTR;
	typedef typename PairType::first_type FT;
	typedef typename PairType::second_type ST;
	typedef typename ::s11n::type_traits<FT>::type FT_base;
	typedef typename ::s11n::type_traits<ST>::type ST_base;

	const NodeType * ch = 0;

	////////////////////////////////////////////////
	// The abstract_creator code below is simply
	// to treat pointer and value types
	// identically with this same code base. See
	// its docs for what it does (or doesn't do,
	// in the case of reference types).

	typedef s11n::Private::abstract_creator<FT> AC1st;
	typedef s11n::Private::abstract_creator<ST> AC2nd;

	using namespace ::s11n::debug;
#define ERRMSG S11N_TRACE(TRACE_ERROR) << "deserialize_pair(node name="<<NTR::name(src)<<"/class="<<NTR::class_name(src)<<", PairType &): "

	//////////////////////////////////////// .first:
	ch = ::s11n::find_child_by_name( src, "first" );
	if( ! ch )
	{
		ERRMSG << "deserialize_pair: deserialize: no 'first' node found!\n";
		return false;
	}
	std::string implclass = NTR::class_name( *ch );
	FT f; // value of .first
	try
	{
		if( ! AC1st::create( f, implclass ) )
		{
			ERRMSG << "Internal error: could not create .first element."
			       <<"type='"<<NTR::class_name( *ch )<<"'!\n";
			return false;
		}
		if( ! ::s11n::deserialize<NodeType,FT>( *ch, f ) )
		{
			::s11n::cleanup_serializable<FT_base>( f );
			ERRMSG << "first deserialize(..., first ) failed.\n";
			return false;
		}
	}
	catch(...) // deser threw
	{
		ERRMSG << "first deserialize() threw. Cleaning up, then propagating it...\n";
		::s11n::cleanup_serializable<FT_base>( f );
		throw;
	}

	//////////////////////////////////////// .second:
	ch = ::s11n::find_child_by_name( src, "second" );
	if( ! ch )
	{
		ERRMSG << "no 'second' node found!\n";
		::s11n::cleanup_serializable<FT_base>( f );
		return false;
	}
	implclass = NTR::class_name( *ch );
	ST s; // value of .second
	try
	{
		if( ! AC2nd::create( s, implclass ) )
		{
			ERRMSG << "Internal error: could not create .second element."
			       <<"type='"<<implclass<<"'!\n";
			::s11n::cleanup_serializable<FT_base>( f );
			return false;
		}
		if( ! s11n::deserialize<NodeType,ST>( *ch, s ) )
		{
			ERRMSG << "deserialize(node, second) failed.\n";
			::s11n::cleanup_serializable<FT_base>( f );
			::s11n::cleanup_serializable<ST_base>( s );
			return false;
		}
	}
	catch(...)
	{
		ERRMSG << "second deserialize() threw. Cleaning up, then propagating it...\n";
		::s11n::cleanup_serializable<FT_base>( f );
		::s11n::cleanup_serializable<ST_base>( s );
		throw;
	}
	dest.first = f;
	dest.second = s;
	return true;
#undef ERRMSG
}


template <typename NodeType, typename MapType>
bool s11n::map::serialize_map( NodeType & dest, const MapType & src )
{
	typedef typename MapType::const_iterator CIT;
	typedef typename MapType::value_type PairType;
	typedef node_traits<NodeType> TR;
	typedef ::s11n::s11n_traits<MapType> STR;
	if( ! TR::empty( dest ) )
	{
		throw ::s11n::s11n_exception( "%s:%d: serialize_map(tgt_node=[%s]) requires that the target node be empty.",
					      __FILE__,
					      __LINE__,
					      TR::name(dest).c_str() );
	}
	TR::class_name( dest, STR::class_name(&src) );
	CIT b = src.begin();
	CIT e = src.end();
	for( ; e != b; ++b )
	{
		std::unique_ptr<NodeType> ch( TR::create("pair") );
		if( ! serialize_pair<NodeType,PairType>( *ch, *b ) )
		{
			using namespace ::s11n::debug;
			S11N_TRACE(TRACE_ERROR) << "serialize_map: child pair failed serialize.\n";
			return false;
		}
		dest.children().push_back( ch.release() ); // transfer ownership
	}
	return true;
}

template <typename NodeType, typename MapType>
bool s11n::map::serialize_map( NodeType & dest,
			       const std::string & subnodename,
			       const MapType & src )
{
	typedef node_traits<NodeType> TR;
	std::unique_ptr<NodeType> ch( TR::create(subnodename) );
	if( ! serialize_map<NodeType,MapType>( *ch, src ) )
	{
		return false;
	}
	TR::children(dest).push_back( ch.release() );
	return true;
}


template <typename NodeType, typename MapType>
bool s11n::map::deserialize_map( const NodeType & src, MapType & dest )
{
	typedef node_traits<NodeType> TR;
	typedef typename NodeType::child_list_type::const_iterator CIT;
	//typedef typename SerializableType::value_type VT;
	// ^^^ no, because VT::first_type is const!
	// Thus we hand-create a compatible pair type:
	typedef typename MapType::key_type KType;
	typedef typename MapType::mapped_type VType;
	typedef std::pair< KType, VType > PairType;
	PairType pair;
	CIT b = TR::children(src).begin();
	CIT e = TR::children(src).end();
	if( e == b ) return true;
	const NodeType *ch = 0;
	using namespace ::s11n::debug;
#define ERRMSG S11N_TRACE(TRACE_ERROR) << "deserialize_map(node,map) srcnode="<<std::dec<<&src << ": "
	MapType buffer;
	try
	{
		for( ; e != b ; ++b )
		{
			ch = *b;
			if( ! deserialize_pair<NodeType,PairType>( *ch, pair ) )
			{ // ^^^ is this fails, pair is now guaranteed to be unmodified (1.1.3)
				ERRMSG << "map child failed deser: node name='"<<TR::name(*ch)<<"'. Cleaning up map.\n";
				::s11n::cleanup_serializable<MapType>( buffer );
				return false;
			}
			buffer.insert( pair );
		}
	}
	catch(...)
	{
		ERRMSG << "Passing on exception. Map cleaned using s11n_traits::cleanup_functor.\n";
		::s11n::cleanup_serializable<MapType>( buffer );
		throw;
	}
#undef ERRMSG
	dest.swap( buffer );
	return true;
}


template <typename NodeType, typename MapType>
bool s11n::map::deserialize_map( const NodeType & src,
				 const std::string & subnodename,
				 MapType & dest )
{
	const NodeType * ch = ::s11n::find_child_by_name( src, subnodename );
	if( ! ch ) return false;
	return deserialize_map<NodeType,MapType>( *ch, dest );
}



#endif // s11n_net_s11n_v1_1_MAP_TPP_INCLUDED
