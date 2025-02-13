#ifndef s11n_net_s11n_std_pair_HPP_INCLUDED
#define s11n_net_s11n_std_pair_HPP_INCLUDED 1

#include <s11n.net/s11n/proxy/mapish.hpp>

#define S11N_TEMPLATE_TYPE std::pair
#define S11N_TEMPLATE_TYPE_NAME "pair"
// #define S11N_TEMPLATE_TYPE_PROXY ::s11n::map::pair_serializable_proxy
#define S11N_TEMPLATE_TYPE_PROXY ::s11n::map::serialize_pair_f
#define S11N_TEMPLATE_TYPE_DESER_PROXY ::s11n::map::deserialize_pair_f
#include <s11n.net/s11n/proxy/reg_s11n_traits_template2.hpp>

namespace s11n {

	template <typename T1, typename T2>
	struct default_cleanup_functor< std::pair<T1,T2> >
	{
		typedef T1 first_type;
		typedef T2 second_type;
		typedef std::pair<T1,T2> serializable_type;
		void operator()( serializable_type & p ) const
		{
			using namespace s11n::debug;
			S11N_TRACE(TRACE_CLEANUP) << "default_cleanup_functor<> specialization cleaning up pair<>...\n";
  			typedef typename type_traits<first_type>::type TR1;
  			typedef typename type_traits<second_type>::type TR2;
			cleanup_serializable<TR1>( p.first );
			cleanup_serializable<TR2>( p.second );

		}
	};
}

#endif // s11n_net_s11n_std_pair_HPP_INCLUDED
