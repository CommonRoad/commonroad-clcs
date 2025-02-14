////////////////////////////////////////////////////////////////////////
// valarray.hpp: s11n algos/proxies for handling std::valarray objects.
// This header is not included by default: clients who want to de/serialize
// valarrays must include it themselves.
////////////////////////////////////////////////////////////////////////
#ifndef s11n_VALARRAY_HPP_INCLUDED
#define s11n_VALARRAY_HPP_INCLUDED 1

#include <stdio.h> // snprintf()
#include <valarray>
#include <s11n.net/s11n/variant.hpp> // for lexical casting

namespace s11n {

        /**
           s11n::va encapsulates de/serialization operators for
           std::valarray objects.
        */
        namespace va {

                /**
                   Serializes src to dest. Returns true on success,
                   false on error. VAT must be a std::valarray type
                   with a numeric value_type.

		   Always returns true.

		   Each entry is stored as a key/value pair in src,
		   which means this algorithm works with all
		   iostreamable contained types. Since valarray is
		   intended for use with numbers, this should be
		   appropriate for all cases.

		   To avoid entering numeric keys into dest, which don't
		   work with XML serializers, and to keep proper ordering of
		   the keys, it synthesizes sequential numbers, preceded by
		   an 'x', for use as keys. The number of digits in the keys
		   is calculated based off of src.size().

		   This function never returns false, but will throw if
		   src has more than some arbitrarily large number of items
		   (at least 64 bits worth), due to a minor detail you can
		   read about in the source code.

		   ACHTUNG: precision of doubles in limited to
		   whatever default is used by s11n for lexical
		   casting via iostreams.
                */
                template <typename NodeT, typename VAT>
                bool serialize_valarray( NodeT & dest, const VAT & src )
                {
                        typedef s11n::node_traits<NodeT> TR;
                        TR::class_name( dest, ::s11n::s11n_traits<VAT>::class_name(&src) );
                        const int buffsize = 9; // leaves us with 8 hex digits, which is more than your system can hold in one valarray, i would think.
                        char num[buffsize];
                        char fmt[buffsize];
                        size_t sz = src.size();
                        int places = 1; // # of digits to use
                        for( ; sz >= 0x0f; sz = (size_t)(sz/0x0f)) { ++places; }
			if( places > (buffsize-1) )
			{
				throw ::s11n::s11n_exception( "%s:%d: Internal error: overflow in serialize_valarray(). Too many items in valarray.", __FILE__,__LINE__);
			}
                        snprintf( fmt, buffsize, "x%%0%dx", places ); // e.g. 5 digits evals to "x%05x"
                        sz = src.size();
                        TR::set( dest, "size", sz );
                        for( size_t i = 0 ; i < sz; i++ )
                        {
                                snprintf( num, buffsize, fmt, i );
                                TR::set( dest, num, src[i] );
                        }
                        return true;
                }

                /**
                   Deserializes dest from src. Returns true on
                   success, false on error. VAT must be a
                   std::valarray type with a numeric value_type.

		   Always returns true.
                */
                template <typename NodeT, typename VAT>
                bool deserialize_valarray( const NodeT & src, VAT & dest )
                {
                        typedef ::s11n::node_traits<NodeT> TR;
                        typedef typename VAT::value_type VT;
                        typename TR::property_map_type::const_iterator it = TR::properties(src).begin();
			typename TR::property_map_type::const_iterator et = TR::properties(src).end();
                        const static std::string szkey = "size";
                        size_t size = TR::get( src, szkey, dest.size() );
                        VT defaultval;
                        dest.resize( size, defaultval );
                        size_t i = 0;
                        for( ; et != it; ++it )
                        {
                                if( szkey == (*it).first ) continue;
                                dest[i++] = ::s11n::Detail::variant( (*it).second ).template cast_to<VT>( defaultval );
                        }
                        return true;
                }

                /**
                   A Serializable proxy for valarray types.
                */
                struct valarray_serializable_proxy
                {

                        valarray_serializable_proxy()
                        {}

                        /**
                           see serialize_valarray().

                        */
                        template <typename NodeType, typename SerType>
                        bool operator()( NodeType & dest, const SerType & src ) const
                        {
                                return serialize_valarray( dest, src );
                        }

                        /** see deserialize_valarray(). */
                        template <typename NodeType, typename SerType>
                        bool operator()( const NodeType & src, SerType & dest ) const
                        {
                                return deserialize_valarray( src, dest );
                        }
                };

        } // namespace va



        /**
           s11n_traits<> specialization for std::valarray types.
        */
        template <typename ValueType>
        struct s11n_traits < std::valarray<ValueType> >
        {
                typedef std::valarray<ValueType> serializable_type;
                typedef ::s11n::va::valarray_serializable_proxy serialize_functor;
                typedef serialize_functor deserialize_functor;
                typedef ::s11n::cl::object_factory<serializable_type> factory_type;
		typedef ::s11n::default_cleanup_functor<serializable_type> cleanup_functor;
                static bool cl_reg_placeholder;
		static std::string class_name( const serializable_type * instance_hint )
		{
			if( cl_reg_placeholder == true ); // just to reference it. w/o this cl reg never happens :(
			return "valarray";
		}
        };

	template <typename VT>
	bool s11n_traits< std::valarray< VT > >::cl_reg_placeholder =
		(
		 ::s11n::cl::classloader_register_base<std::valarray< VT > >( s11n_traits< std::valarray< VT > >::class_name(0) )
		 , 0 );


} // namespace s11n



#endif // s11n_VALARRAY.HPP_HPP_INCLUDED
