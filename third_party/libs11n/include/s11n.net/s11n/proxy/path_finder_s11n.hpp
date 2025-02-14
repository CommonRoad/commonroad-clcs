#ifndef s11n_net_s11n_PROXY_PATH_FINDER_HPP_INCLUDED
#define s11n_net_s11n_PROXY_PATH_FINDER_HPP_INCLUDED 1

#include <s11n.net/s11n/plugin/path_finder.hpp>
#include <s11n.net/s11n/proxy/listish.hpp>

namespace s11n { namespace plugin {

	/**
	   An s11n proxy for s11n::plugin::path_finder.
	*/
	struct path_finder_s11n
	{
		// serialize
		template <typename NodeType>
		bool operator()( NodeType & dest, const path_finder & src ) const
		{
			typedef ::s11n::node_traits<NodeType> TR;
			if( ! ::s11n::list::serialize_streamable_list(  dest, "path", src.path() ) ) return false;
			if( ! ::s11n::list::serialize_streamable_list(  dest, "extensions", src.extensions() ) ) return false;
			return true;
		}
		
		// deserialize
		template <typename NodeType>
		bool operator()( const NodeType & src, path_finder & dest ) const
		{
			typedef ::s11n::node_traits<NodeType> TR;
			if( ! ::s11n::list::deserialize_streamable_list( src, "path", dest.path() ) )return false;
			if( ! ::s11n::list::deserialize_streamable_list( src, "extensions", dest.extensions() ) ) return false;
			return true;
		}

	}; // path_finder_s11n

}} // namespaces

#define S11N_TYPE s11n::plugin::path_finder
#define S11N_TYPE_NAME "s11n::plugin::path_finder"
#define S11N_SERIALIZE_FUNCTOR s11n::plugin::path_finder_s11n
#include <s11n.net/s11n/reg_s11n_traits.hpp>
#endif // s11n_net_s11n_PROXY_PATH_FINDER_HPP_INCLUDED
