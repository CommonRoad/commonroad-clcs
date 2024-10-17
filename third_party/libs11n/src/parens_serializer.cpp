
#include <s11n.net/s11n/s11n_node.hpp>
#include <s11n.net/s11n/io/serializers.hpp>
#include <s11n.net/s11n/phoenix.hpp>
#include <s11n.net/s11n/io/parens_serializer.hpp>

namespace s11n { namespace io {


        /**
           Internal-use initializer for setting up a Serializer
           translation map.
        */
        struct paren_serializer_translations_initializer
        {
                template <typename MapT>
                void operator()( MapT & map )
                {
                        // The order of these escapes is
                        // signifant.  We only do
                        // double-backslashes to accomodate
                        // the case that the final chars in a
                        // property is a backslash (yes, this
                        // has happened, and it hosed the
                        // data, because it inadvertently
                        // escaped a control token.).
                        map["\\"] = "\\\\";
                        map[")"] = "\\)";
                        map["("] = "\\(";
                        // It is not strictly necessary to escape \(,
                        // but we do so because Parens is intended to
                        // be easy for hand-editing, and not escaping
                        // them confuses emacs when we have escaped
                        // closing parens. :)
                }
        };

        /**
           Returns the translations map for parens_serializer.
        */
        entity_translation_map &
        parens_serializer_translations()
        {
                typedef s11n::Detail::phoenix<
                entity_translation_map,
                        sharing::parens_sharing_context,
                        paren_serializer_translations_initializer
                        > TheMap;
                return TheMap::instance();
        }


} } // namespace s11n::io


namespace {

        void parens_serializer_registration_init()
        {

#define SERINST(NodeT)                                                  \
                ::s11n::io::register_serializer< ::s11n::io::parens_serializer< NodeT > >( "s11n::io::parens_serializer", "parens" );

                SERINST(s11n::s11n_node);
#undef SERINST
        }

	// will be called during static init phase, triggering serializer setup
        int parens_reg_placeholder = 
                ( parens_serializer_registration_init(), 1 );
} // anonymous namespace
