
#include <s11n.net/s11n/s11n_node.hpp>
#include <s11n.net/s11n/phoenix.hpp>
#include <s11n.net/s11n/io/serializers.hpp>
#include <s11n.net/s11n/io/funtxt_serializer.hpp>
#include <s11n.net/s11n/io/FlexLexer.hpp>

namespace s11n { namespace io {


        /**
           Internal-use initializer for setting up a Serializer
           translation map.
        */
        struct funtxt_serializer_translations_initializer
        {
                void operator()( entity_translation_map & map )
                {
                        // the order of these escapes is signifant.
                        // We only do double-backslashes to accomodate
                        // the case that the final char in a property
                        // is a backslash (yes, this has happened, and
                        // it hosed the input).
                        map["\\"] = "\\\\";
                        map["\n"] = "\\\n";
                        map["{"] = "\\{";
                        map["}"] = "\\}";
                }
        };

        /**
           Returns the translations map for funtxt_serializer.
        */
        entity_translation_map &
        funtxt_serializer_translations()
        {
                typedef ::s11n::Detail::phoenix<entity_translation_map,
                        sharing::funtxt_sharing_context,
                        funtxt_serializer_translations_initializer
                        > TMap;
                return TMap::instance();
        }

}} // namespace s11n::io

namespace {

        void funtxt_serializer_registration_init()
        {

#define SERINST(NodeT)                                  \
                ::s11n::io::register_serializer< ::s11n::io::funtxt_serializer< NodeT > >( "s11n::io::funtxt_serializer", "funtxt" );
                SERINST(s11n::s11n_node);
#undef SERINST
        }

        int funtxt_reg_placeholder = 
                ( funtxt_serializer_registration_init(), 1 );



} // anonymous namespace
