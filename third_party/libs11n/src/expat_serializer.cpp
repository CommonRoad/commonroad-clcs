
#include <s11n.net/s11n/phoenix.hpp>

#include <s11n.net/s11n/s11n_node.hpp>
#include <s11n.net/s11n/io/serializers.hpp>
#include <s11n.net/s11n/io/expat_serializer.hpp>

namespace s11n { namespace io {


        /**
           Internal-use initializer for setting up the expat
           Serializer translation map.
        */
        struct expat_serializer_translations_initializer
        {
                template <typename MapType>
                void operator()( MapType & map )
                {
                        map["&"] = "&amp;";
                        map["\""] = "&quot;";
                        map["'"] = "&apos;";
                        map[">"] = "&gt;";
                        map["<"] = "&lt;";
                }
        };


        entity_translation_map &
        expat_serializer_translations()
        {
                typedef ::s11n::Detail::phoenix<
                entity_translation_map,
                        sharing::expat_sharing_context,
                        expat_serializer_translations_initializer
                        > TheMap;
                return TheMap::instance();
        }

} } // namespace s11n::io

namespace {

        void expat_serializer_registration_init()
        {

#define SERINST(NodeT)                                  \
                ::s11n::io::register_serializer< ::s11n::io::expat_serializer< NodeT > >( "s11n::io::expat_serializer", "expat" );
                SERINST(s11n::s11n_node);
#undef SERINST
        }

        int expat_reg_placeholder = 
                ( expat_serializer_registration_init(), 1 );



} // anonymous namespace
