#ifndef s11n_net_s11n_v1_1_LIST_TPP_INCLUDED
#define s11n_net_s11n_v1_1_LIST_TPP_INCLUDED
// Implementation code for interfaces declared in listish.hpp

#include <stdio.h> // snprintf()

#include <list>
#include <vector>
#include <set>
#include <memory>
#include <iterator> // insert_iterator
#include <algorithm> // for_each()

#include <s11n.net/s11n/variant.hpp> // lexical casting
#include <s11n.net/s11n/traits.hpp> // node_traits<>
#include <s11n.net/s11n/type_traits.hpp> // type_traits<>
#include <s11n.net/s11n/serialize.hpp> // core serialize funcs
#include <s11n.net/s11n/abstract_creator.hpp> // abstract_creator class
#include <s11n.net/s11n/exception.hpp> // s11n_exception class

#include <s11n.net/s11n/s11n_debuggering_macros.hpp> // tracing macros


template <typename NodeType, typename SerType>
bool s11n::list::serialize_list( NodeType & dest, const SerType & src )
{
	typedef ::s11n::node_traits<NodeType> TR;
	if( ! TR::empty( dest ) )
	{
		throw ::s11n::s11n_exception( "%s:%d: serialize_list() requires that the target node be empty.", __FILE__, __LINE__ );
	}
	typedef ::s11n::s11n_traits<SerType> STR;
	typedef typename SerType::value_type VT;
	TR::class_name( dest, STR::class_name(&src) );
	typename SerType::const_iterator it = src.begin();
	// reminder: (*it) may be a pointer- or a value-type,
	// as long as it's a Serializable.
	using namespace ::s11n::debug;
	for( ; src.end() != it; ++it )
	{
		std::unique_ptr<NodeType> ch( TR::create() );
		if( ::s11n::serialize<NodeType,VT>( *ch, *it ) )
		{
			TR::children(dest).push_back( ch.release() );
			continue;
		}
		S11N_TRACE(TRACE_ERROR) << "serialize_list: a child failed to serialize: "
					<< TR::name(*ch) << " @ " << std::hex << ch.get() << "\n";
		return false;
	}
	return true;
}

template <typename NodeType, typename SerType>
bool s11n::list::serialize_list( NodeType & dest,
				 const std::string & subnodename,
				 const SerType & src )
{
	typedef node_traits<NodeType> TR;
	std::unique_ptr<NodeType> ch( TR::create(subnodename) );
	if( serialize_list<NodeType,SerType>( *ch, src ) )
	{
		TR::children(dest).push_back( ch.release() );
		return true;
	}
	return false;
}



template <typename NodeType, typename SerType>
bool s11n::list::deserialize_list( const NodeType & src, SerType & dest )
{
	// note: some (most) of the odd-looking code
	// here is to allow this func to support both
	// pointer- and non-pointer value_types in SerType.
	// The other half is debug and error handling.
	typedef typename SerType::value_type VT;
	typedef s11n::Private::abstract_creator<VT> ACVT;
	typedef typename NodeType::child_list_type::const_iterator CHIT;
	typedef node_traits<NodeType> TR;

	const NodeType * nch = 0;
	CHIT it = TR::children(src).begin();
	CHIT et = TR::children(src).end();
	if( et == it ) return true;
	VT ser; // reminder: might be a pointer type
	typedef typename ::s11n::type_traits<VT>::type VT_base;
	std::string implclass;
	using namespace ::s11n::debug;
	SerType buffer;
#define ERRMSG S11N_TRACE(TRACE_WARNING) << "deserialize_list(node,list) srcnode="<<std::dec<<&src << ": "
	for( ; et != it; ++it )
	{
		nch = *it;
		if( ! nch /* this never happens */ )
		{
			ERRMSG << "problemus internus: got a null child entry. "
			       << "Throwing and leaving list in its current state.\n";
			throw ::s11n::s11n_exception("%s:%d: Internal error: node children list contains a null pointer.", __FILE__, __LINE__ );
			return false;
		}
		implclass = TR::class_name(*nch);
		// instantiate a new child object to deserialize to...
		try
		{
			if( ! ACVT::create( ser, implclass ) ) // might throw
			{
				ERRMSG << "Internal error: abstract_creator<> "
				       << "could not create a new object of type '"
				       << implclass<<"'!\n";
				::s11n::cleanup_serializable<SerType>( buffer );
				return false;
			}
		}
		catch(...)
		{
			ERRMSG << "abstract_creator::create(...,"<<implclass<<") threw! Cleaning up and passing on the exception.\n";
			::s11n::cleanup_serializable<SerType>( buffer );
			throw;
		}
		// populate the child...
		try
		{
			// CERR << "This really works on pointers, right?\n";
			if( ! ::s11n::deserialize<NodeType,VT_base>( *nch, ser ) ) // might throw
			{
				ERRMSG << "deserialize_list(): deser of a child failed!\n"
				       << "name="<< TR::name(*nch)<< ". implclass="<< implclass
				       <<" @ " << std::hex<<nch <<"\n";
				::s11n::cleanup_serializable<VT_base>( ser );
				::s11n::cleanup_serializable<SerType>( buffer );
				return false;
			}
		}
		catch(...)
		{
			ERRMSG << "deserialize_list() got exception. Using cleanup_functor on failed child, then on container.\n";
			::s11n::cleanup_serializable<VT_base>( ser );
			::s11n::cleanup_serializable<SerType>( buffer );
			throw;
		}
		// add it to our list
		buffer.insert( buffer.end(), ser );
	} // for()
	
#undef ERRMSG
	dest.swap( buffer );// move buffer contents into dest list
	return true;
}

template <typename NodeType, typename SerType>
bool s11n::list::deserialize_list( const NodeType & src,
				   const std::string & subnodename,
				   SerType & dest )
{
	const NodeType * ch = s11n::find_child_by_name( src, subnodename );
	if( ! ch ) return false;
	return deserialize_list<NodeType,SerType>( *ch, dest );
}        


template <typename NodeType, typename ListType>
bool s11n::list::serialize_streamable_list( NodeType & dest, const ListType & src )
{
	typedef typename ListType::const_iterator CIT;
	typedef node_traits<NodeType> TR;
	typedef ::s11n::s11n_traits<ListType> STR;
	TR::class_name( dest, STR::class_name(&src) );
	size_t i = 0;
	// We zero-pad all keys to be the same length, so that
	// they will be kept in the proper order in the
	// target's property list (annoying, but necessary).
	// We prepend a non-numeric char to make this output
	// compatible with standard XML
	const int bsize = 10;
	char num[bsize];
	char fmt[bsize];
	size_t sz = src.size();
	int places = 1; // # of digits to use
	for( ; sz >= 0x0f; sz = (size_t)(sz/0x0f)) { ++places; }
	snprintf( fmt, bsize, "x%%0%dx", places );
	CIT it = src.begin();
	for( ; src.end() != it; ++it )
	{
		snprintf( num, bsize, fmt, i );
		++i;
		TR::set( dest, num, (*it) );
	}
	return true;
}


template <typename NodeType, typename ListType>
bool s11n::list::serialize_streamable_list( NodeType & dest,
					    const std::string & subnodename,
					    const ListType & src )
{
	NodeType & n = s11n::create_child( dest, subnodename );
	return serialize_streamable_list<NodeType,ListType>( n, src );
}


template <typename NodeType, typename ListType>
bool s11n::list::deserialize_streamable_list( const NodeType & src, ListType & dest )
{
	typedef node_traits<NodeType> TR;
	typedef typename ListType::value_type VT;
	typedef typename TR::property_map_type PMT;
	typedef typename PMT::const_iterator CIT;
	CIT it = TR::properties(src).begin();
	CIT et = TR::properties(src).end();
	VT defaultval;
	for( ; et != it; ++it )
	{
		dest.insert( dest.end(),
			     ::s11n::Detail::variant( (*it).second ).template cast_to<VT>( defaultval ) );
	}
	return true;
}


template <typename NodeType, typename ListType>
bool s11n::list::deserialize_streamable_list( const NodeType & src,
					      const std::string & subnodename,
					      ListType & dest )
{
	const NodeType * ch = s11n::find_child_by_name( src, subnodename );
	if( ! ch ) return false;
	return deserialize_streamable_list<NodeType,ListType>( *ch, dest );
}


#endif // s11n_net_s11n_v1_1_LIST_TPP_INCLUDED

