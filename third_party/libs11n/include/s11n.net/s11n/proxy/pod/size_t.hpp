#ifndef s11n_net_s11n_pod_size_t_HPP_INCLUDED
#define s11n_net_s11n_pod_size_t_HPP_INCLUDED 1

// Be careful with this proxy because on 64-bit architectures
// size_t is normally the same as (unsigned long). That means
// that if you register both size_t and ulong on those platforms,
// you'll get ODR violations.
#define S11N_TYPE size_t
#define S11N_TYPE_NAME "size_t"
#define S11N_SERIALIZE_FUNCTOR s11n::streamable_type_serialization_proxy
#include <s11n.net/s11n/reg_s11n_traits.hpp>


#endif // s11n_net_s11n_pod_size_t_HPP_INCLUDED
