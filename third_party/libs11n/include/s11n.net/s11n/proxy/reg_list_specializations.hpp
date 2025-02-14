////////////////////////////////////////////////////////////////////////
// A supermacro to generate some partial template specializations for
// s11n-proxying list-type classes.  See the various proxy files, like
// std/list.hpp, for how to use it.
////////////////////////////////////////////////////////////////////////
#ifndef S11N_LIST_TYPE
#error "You must define S11N_LIST_TYPE before including this file. e.g., to std::list or std::vector."
#endif

#ifndef S11N_LIST_TYPE_NAME
#error "You must define S11N_LIST_TYPE_NAME before including this file. e.g., to \"list\" or \"vector\"."
#endif

#ifndef S11N_LIST_TYPE_PROXY
#define S11N_LIST_TYPE_PROXY ::s11n::list::serialize_list_f
#endif

#ifndef S11N_LIST_TYPE_DESER_PROXY
#  define S11N_LIST_TYPE_DESER_PROXY ::s11n::list::deserialize_list_f
#endif


#include <algorithm> // for_each()
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
		   Calls s11n::cleanup_serializable() for each
		   item in [p.begin(),p.end()).
		*/
		void operator()( serializable_type & p ) const throw()
		{
			std::for_each( p.begin(), p.end(), ::s11n::cleaner_upper() );
			p.clear();
			// well, that was easy.
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
