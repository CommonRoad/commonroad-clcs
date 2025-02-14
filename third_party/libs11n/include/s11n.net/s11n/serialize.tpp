#ifndef s11n_SERIALIZE_TPP_INCLUDED
#define s11n_SERIALIZE_TPP_INCLUDED
// this is the implementation file for the code declared in serialize.hpp


	/**
	********************************************************************
	General API Conventions:

	NodeType should conform to the conventions laid out
	by s11n::s11n_node.

	SerializableTypes/BaseTypes:

	- BaseT must have the following in it's interface:

	- bool SerializeFunction( NodeType & dest ) const;

	- bool DeserializeFunction( const NodeType & dest );

	SerializeFunction/DeserializeFunction need not be virtual,
	though they may be.

	Proxy functors:

	Serialization functor must have:

	bool operator()( NodeType & dest, const BaseT & src ) const;

	Deserialization functor must have:

	bool operator()( const NodeType & src, BaseT & dest ) const;

	They may be the same functor type - const resolution will
	determine which s11n uses. Sometimes this might cause an
	ambiguity, and may require 2 functors.

	These signatures apply for all functors designed to work as
	de/serialization proxies.

	*********************************************************************
	*/


#include <string>
#include <memory> // auto_ptr

#include <s11n.net/s11n/s11n_debuggering_macros.hpp> // COUT/CERR
#include <s11n.net/s11n/classload.hpp> // classload()
#include <s11n.net/s11n/traits.hpp> // s11n_traits
#include <s11n.net/s11n/type_traits.hpp> // type_traits
#include <s11n.net/s11n/exception.hpp> // s11n_exception and friends

////////////////////////////////////////////////////////////////////////////////
// NO DEPS ON s11n_node.hpp ALLOWED!
////////////////////////////////////////////////////////////////////////////////


template <typename SerializableType>
void s11n::cleanup_serializable( SerializableType & s ) throw()
{
	try
	{
		typename s11n::s11n_traits<SerializableType>::cleanup_functor cf;
		cf(s);
	}
	catch(...)
	{
		using namespace ::s11n::debug;
		S11N_TRACE(TRACE_ERROR) << "Exception thrown during cleanup! INGORING IT! This might mean a mem leak has occurred!\n";
	}
}

template <typename SerializableType>
void s11n::cleanup_serializable( SerializableType * & s ) throw()
{
	using namespace ::s11n::debug;
	S11N_TRACE(TRACE_CLEANUP) << "cleanup_serializable(*&): @ " << std::hex << s << ", s11n_class="<<s11n_traits<SerializableType>::class_name(s)<<"\n";
	if( s )
	{
		s11n::cleanup_serializable<SerializableType>( *s );
		delete s;
		s = 0;
	}
}



template <typename NodeType,typename SerializableType>
bool s11n::Detail::s11n_api_marshaler<NodeType,SerializableType>::serialize( NodeType &dest, const SerializableType & src )
{
	typedef ::s11n::node_traits<node_type> NTR;
	typedef ::s11n::s11n_traits<serializable_type> STR;
	NTR::class_name( dest, STR::class_name(&src) );
	typename STR::serialize_functor sf;
	return sf( dest, src );
}

template <typename NodeType,typename SerializableType>
bool s11n::Detail::s11n_api_marshaler<NodeType,SerializableType>::deserialize( const NodeType & src, SerializableType & dest )
{
	typedef ::s11n::s11n_traits<serializable_type> STR;
	typename STR::deserialize_functor df;
	return df( src, dest );
}


template <typename NodeType,typename SerializableType>
bool s11n::Detail::s11n_api_marshaler<NodeType,SerializableType *>::serialize( NodeType &dest, const SerializableType * const & src )
{
	if( ! src )
	{
		return false;
	}
	return parent_type::serialize( dest, *src );
}

template <typename NodeType,typename SerializableType>
bool s11n::Detail::s11n_api_marshaler<NodeType,SerializableType *>::deserialize( const NodeType & src, SerializableType * & dest )
{
	if( ! dest )
	{
		return false;
	}
	return parent_type::deserialize( src, *dest );
}



template <typename DataNodeType, typename SerializableType>
bool s11n::serialize( DataNodeType & target, const SerializableType & src )
{
	return s11n::Detail::s11n_api_marshaler<DataNodeType,SerializableType>::serialize( target, src );
}


template <typename DataNodeType, typename DeserializableT>
bool s11n::deserialize( const DataNodeType & src, DeserializableT & target )
{
	return s11n::Detail::s11n_api_marshaler<DataNodeType,DeserializableT>::deserialize( src, target );
}

template <typename DataNodeType, typename DeserializableT>
bool s11n::deserialize( const DataNodeType & src, DeserializableT * & target )
{
	using namespace ::s11n::debug;
	S11N_TRACE(TRACE_INFO) << "using experimental deserialize(const node &, DeserT * &)\n";
	typedef s11n::node_traits<DataNodeType> NTR;
	typedef s11n::s11n_traits<DeserializableT> STR;
	if( target )
	{
		return s11n::Detail::s11n_api_marshaler<DataNodeType,DeserializableT>::deserialize( src, *target );
	}
	// We want to destroy the obj we allocate if deser fails:
	typedef s11n::cleanup_ptr<DeserializableT> CP;
	CP x( ::s11n::cl::classload<DeserializableT>( NTR::class_name( src ) ) );
	if( ! x.get() ) // try harder...
	{
		// S11N_TRACE(TRACE_FACTORY_LOOKUP) << "Trying to dyn-load obj of class '"<<STR::class_name( target )<<"'\n";
		x.take( ::s11n::cl::classload<DeserializableT>( STR::class_name( target /** 0 ?? **/ ) ) );
	}
	if( ! x.get() ) // give up :(
	{
		S11N_TRACE(TRACE_ERROR) << "deserialize<NT,ST>(const NT &, ST * &): dyn-load failed for class '"<<NTR::class_name( src )<<"'\n";
		return 0;
	}
	if( ! Detail::s11n_api_marshaler<DataNodeType,DeserializableT>::deserialize( src, *x ) )
	{
		return false;
	}
	target = x.release();
	return true;
}

template <typename DataNodeType, typename DeserializableT>
bool s11n::deserialize( const DataNodeType & src, s11n::cleanup_ptr<DeserializableT> & target )
{
	DeserializableT * d = target.get();
	if( deserialize<DataNodeType,DeserializableT>( src, d ) )
	{
		target.take(d);
		// ^^^^^^^^ We actually only have to take(d) only if d
		// started out as 0, but why bother with a flag?
		// Remeber also that take() explicitely does nothing
		// if (get()==d), which means we don't need to
		// duplicate that check here. The case that take(d) is
		// passed 0 AND (d != 0) at call time cannot happen
		// here, so we need not worry about stomping an
		// existing object with take(d).
		return true;
	}
	return false;
}


template <typename DataNodeType, typename DeserializableT>
DeserializableT * s11n::deserialize( const DataNodeType & src )
{
	typedef typename s11n::type_traits<DeserializableT>::type stype;
	typedef ::s11n::s11n_traits<stype> STR;
	typedef ::s11n::node_traits<DataNodeType> NTR;
	typedef typename STR::factory_type FAC;
	s11n::cleanup_ptr<stype> obj( FAC()( NTR::class_name( src ) ) );
	if( ! obj.get() )
	{
		using namespace ::s11n::debug;
		S11N_TRACE(TRACE_ERROR)
			<< "deserialize<>(DataNode): classload failed for class '"
			<< NTR::class_name( src )<<"'."
			<< " It is probably not registered with its base-most classloader.\n";
		return 0;
	}
	try
	{
		if( s11n::deserialize<DataNodeType,stype>( src, *obj ) ) // might throw
		{
			return obj.release(); // take over ownership.
		}
	}
	catch( ... )
	{
		using namespace ::s11n::debug;
		S11N_TRACE(TRACE_ERROR)
			<< "deserialize<NT,DeserT>(const NT&) failed deser'ing child and is using cleanup_serializable() on it.\n";
		throw;
	}
	return 0;
}

template <typename DataNodeType, typename SerializableType>
SerializableType * s11n::s11n_clone( const SerializableType & tocp )
{
	DataNodeType node;
	return ::s11n::serialize<DataNodeType,SerializableType>( node, tocp )
		? ::s11n::deserialize<DataNodeType,SerializableType>( node )
		: 0;
}

template <typename NodeType, typename Type1, typename Type2>
bool s11n::s11n_cast( const Type1 & t1, Type2 & t2 )
{
	NodeType n;
	return ::s11n::serialize<NodeType,Type1>( n, t1 ) && ::s11n::deserialize<NodeType,Type2>( n, t2 );
}

#endif // s11n_SERIALIZE_TPP_INCLUDED
