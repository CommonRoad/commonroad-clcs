#ifndef s11n_net_s11n_std_set_HPP_INCLUDED
#define s11n_net_s11n_std_set_HPP_INCLUDED 1

#include <s11n.net/s11n/proxy/listish.hpp>

#define S11N_LIST_TYPE std::multiset
#define S11N_LIST_TYPE_NAME "multiset"

#ifndef S11N_LIST_TYPE_PROXY
#define S11N_LIST_TYPE_PROXY ::s11n::list::serialize_list_f
#endif

#ifndef S11N_LIST_TYPE_DESER_PROXY
#  define S11N_LIST_TYPE_DESER_PROXY ::s11n::list::deserialize_list_f
#endif


namespace s11n {

	/**
	   Specialization for list-compliant types.
	*/
	template < typename T1 >
	struct S11N_EXPORT_API default_cleanup_functor< S11N_LIST_TYPE< T1 > >
	{
		typedef T1 value_type;
		typedef S11N_LIST_TYPE< T1 > serializable_type;

		/**
		   Do nothing. In a multiset<T>, the contained members are
		   const, so we can't rightfully do anything about
		   them here.
		*/
		void operator()( serializable_type & p ) const throw()
		{
		}
	};

} // namespace


#define S11N_TEMPLATE_TYPE S11N_LIST_TYPE
#define S11N_TEMPLATE_TYPE_NAME S11N_LIST_TYPE_NAME
#define S11N_TEMPLATE_TYPE_PROXY S11N_LIST_TYPE_PROXY
#define S11N_TEMPLATE_TYPE_DESER_PROXY S11N_LIST_TYPE_DESER_PROXY
#include <s11n.net/s11n/proxy/reg_s11n_traits_template1.hpp>
#undef S11N_LIST_TYPE_DESER_PROXY
#undef S11N_LIST_TYPE_PROXY
#undef S11N_LIST_TYPE_NAME
#undef S11N_LIST_TYPE


#endif // s11n_net_s11n_std_set_HPP_INCLUDED
