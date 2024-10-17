
#include <s11n.net/s11n/s11n_node.hpp>
#include <s11n.net/s11n/phoenix.hpp>
#include <s11n.net/s11n/io/serializers.hpp>
#include <s11n.net/s11n/io/wesnoth_serializer.hpp>

namespace s11n { namespace io {


        /**
           Internal-use initializer for setting up a Serializer
           translation map.
        */
        struct wesnoth_serializer_translations_initializer
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
                        map["\""] = "\\\"";
                        map["["] = "\\[";
                        map["]"] = "\\]";
                }
        };

        /**
           Returns the translations map for wesnoth_serializer.
        */
        entity_translation_map &
        wesnoth_serializer_translations()
        {
                typedef ::s11n::Detail::phoenix<entity_translation_map,
                        sharing::wesnoth_sharing_context,
                        wesnoth_serializer_translations_initializer
                        > TMap;
                return TMap::instance();
        }

}} // namespace s11n::io

namespace {

        void wesnoth_serializer_registration_init()
        {
                //CERR << "wesnoth_serializer_registration_init()\n";
#define SERINST(NodeT)                                  \
                ::s11n::io::register_serializer< ::s11n::io::wesnoth_serializer< NodeT > >( "s11n::io::wesnoth_serializer", "wesnoth" );

                SERINST(s11n::s11n_node);
#undef SERINST
        }

	// will be called during static init phase, triggering serializer setup
        int wesnoth_reg_placeholder = 
                ( wesnoth_serializer_registration_init(), 1 );



} // anonymous namespace
