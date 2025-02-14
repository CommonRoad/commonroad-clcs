
#include <s11n.net/s11n/s11n_node.hpp>
#include <s11n.net/s11n/phoenix.hpp>
#include <s11n.net/s11n/io/serializers.hpp>
#include <s11n.net/s11n/io/simplexml_serializer.hpp>

namespace s11n { namespace io {



        /**
           Internal-use initializer for setting up a Serializer
           translation map.
        */
        struct simplexml_serializer_translations_initializer
        {
                void operator()( entity_translation_map & map )
                {
                        // just the basics are covered here...
                        map["&"] = "&amp;";
                        map["\n"] = "&#010;";
                        map["\""] = "&quot;";
                        map["'"] = "&apos;";
                        map[">"] = "&gt;";
                        map["<"] = "&lt;";
                }
        };

        /**
           Returns the translations map for simplexml_serializer.
        */
        entity_translation_map &
        simplexml_serializer_translations()
        {
                typedef s11n::Detail::phoenix<entity_translation_map,
                        sharing::simplexml_sharing_context,
                        simplexml_serializer_translations_initializer
                        > TMap;
                return TMap::instance();
        }

}} // namespace s11n::io


namespace {

        void simplexml_serializer_registration_init()
        {

#define SERINST(NodeT)                                  \
                ::s11n::io::register_serializer< ::s11n::io::simplexml_serializer< NodeT > >( "s11n::io::simplexml_serializer", "simplexml" );

                SERINST(s11n::s11n_node);
#undef SERINST
        }

	// will be called during static init phase, triggering serializer setup
        int simplexml_reg_placeholder = 
                ( simplexml_serializer_registration_init(), 1 );

} // anonymous namespace
