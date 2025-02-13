
#include <s11n.net/s11n/s11n_node.hpp>
#include <s11n.net/s11n/io/serializers.hpp>
#include <s11n.net/s11n/io/compact_serializer.hpp>

namespace {

        void compact_serializer_registration_init()
        {

#define SERINST(NodeT)                                  \
                ::s11n::io::register_serializer< ::s11n::io::compact_serializer< NodeT > >( "s11n::io::compact_serializer", "compact" );

                SERINST(s11n::s11n_node);
#undef SERINST
        }

        int compact_reg_placeholder = 
                ( compact_serializer_registration_init(), 1 );



} // anonymous namespace
