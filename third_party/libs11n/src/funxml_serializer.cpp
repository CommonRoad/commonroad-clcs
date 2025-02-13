
#include <s11n.net/s11n/s11n_node.hpp>
#include <s11n.net/s11n/phoenix.hpp>
#include <s11n.net/s11n/io/serializers.hpp>
#include <s11n.net/s11n/io/funxml_serializer.hpp>

namespace s11n { namespace io {


        /**
           Internal-use initializer for setting up the funxml
           Serializer translation map.
        */
        struct funxml_serializer_translations_initializer
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
        funxml_serializer_translations()
        {
                typedef s11n::Detail::phoenix<
                entity_translation_map,
                        sharing::funxml_sharing_context,
                        funxml_serializer_translations_initializer
                        > TheMap;
                return TheMap::instance();
        }

} } // namespace s11n::io

namespace {

        void funxml_serializer_registration_init()
        {

#define SERINST(NodeT)                                  \
                ::s11n::io::register_serializer< ::s11n::io::funxml_serializer< NodeT > >( "s11n::io::funxml_serializer", "funxml" );

                SERINST(s11n::s11n_node);
#undef SERINST
        }

	// will be called during static init phase, triggering serializer setup
        int funxml_reg_placeholder = 
                ( funxml_serializer_registration_init(), 1 );


} // anonymous namespace
