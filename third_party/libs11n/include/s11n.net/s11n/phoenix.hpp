#ifndef s11n_net_s11n_PHOENIX_HPP_INCLUDED
#define s11n_net_s11n_PHOENIX_HPP_INCLUDED 1
////////////////////////////////////////////////////////////////////////////////
// phoenix.hpp
// phoenix<> provides "context singletons" with "phoenixing" capabilities.
//
// Author: stephan beal <stephan@s11n.net>
// License: Public Domain
// CVS Revision: $Revision: 1.4 $
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h> // atexit()
#include <iostream> // cout/cerr


#ifndef phoenix_DEBUG
// enable debuggering to see when phoenixes are (re)created.
#    define phoenix_DEBUG 0
#endif

#if phoenix_DEBUG
#    include <typeinfo>
#    define phoenix_CERR std::cerr << __FILE__ << ":" << std::dec << __LINE__ << " : " \
	<< "phoenix<"<<typeid((base_type *)NULL).name()<<" , " \
	<< typeid((context_type *)NULL).name()<<"> "
#else
#    define phoenix_CERR if(0) std::cerr
#endif // phoenix_DEBUG

namespace s11n { namespace Detail {

        
        /**
           Internal helper class to provide a default no-op
           initializer for phoenixed objects.

           See the phoenix<> class.
        */
        struct no_op_phoenix_initializer
        {
                /** Does nothing: This class is called no_op for a reason ;) */
                template <typename T>
                void operator()( T & ) { return; }
        };

        /**

	The phoenix class acts as a wrapper for adding "phoenixing"
	behaviour to arbitrary shared objects, as covered in detail in
	Alexandrescu's "Modern C++ Design".

	phoenix is a class for holding singleton-style instances of
	BaseType objects. Rather than requiring that BaseType be a
	Singleton type, phoenix subclasses BaseType to add the
	phoenix-like capabilities. Phoenixing makes the shared
	object post-main() safe, in terms of object destruction
	order.
	
           Parameterized on:

           - BaseType: must be struct or class type and must be
           default-constructable. i have no clue what is supposed to
           happen if BaseType's dtor is not virtual. That said,
           phoenix has been successfully demonstrated with a BaseType
           of std::map, which has no virtual dtor.

           - ContextType: These objects are only singletons within the
           given ContextType.  That is, phoenix&lt;T,X&gt;::instance()
           will return a different object than phoenix&lt;T,Y&gt;
           will.

           - InitializerType: must be a unary functor accepting a
           BaseType &. It's return value is ignored. The default
           functor does nothing.  The InitializerType is called when a
           to-be-phoenixed object is initially created and whenever it
           is phoenixed. This is intended to be used, e.g., for
           re-populating a phoenixed shared object.
           TODO: investigate the implications of a predicate
           initializer, which would return false if the object could
           not be initialized.


           Whether or not BaseType is technically a singleton depends
           on entirely BaseType itself. This class is more often used
           to provide easy access to context-dependent shared objects,
           rather than pure singletons. The phoenix class itself is a
           true Singleton, but each combination of template arguments
           provides a different Singleton *type*, so the end effect is
           "context singletons."


           This is another attempt to solve the classic
           Keyboard-Console-Log problem, as discussed at length in
           <i>Modern C++ Design</i>. It relies on sane behaviour in
           the C library's atexit() function, which, as is shown in
           MC++D, is not the case on all systems. That said, the
           phoenix-specific behaviours are undefined on those systems,
           which is only to say that it might not be post-main() safe.


           Caveats:

           i am not 100% clear on all of the implications of this
           implementation's approach... my gut tells me i'm missing
           some significant bits. i mean, it <i>can't</i> have been
           this straightforward to solve ;). The very nature of the
           Phoenix Singleton problem makes it difficult to reliably
           test in real-world applications. That said, i have seen a
           objects be successfully phoenixed and atexit()ed, so it is
           known to at least "basically" work.

           There's a paper about "context singletons", this class,
           and some of it's implications, at:

           http://s11n.net/papers/


           [Much later: i've gotten more re-use out of this class than
           probably any other single class i've ever written.]
        */
        template <
                typename BaseType,
                typename ContextType = BaseType,
                typename InitializerType = no_op_phoenix_initializer
        >
        struct phoenix : public BaseType
        {
                /**
                   context_type is unused by this class, but might be useful
                   for type identification at some point.
                */
                typedef ContextType context_type;
                /**
                   The BaseType parameterized type.
                 */
                typedef BaseType base_type;

                /**
                   The functor type used to initialize this phoenixed object.
                */
                typedef InitializerType initializer_type;

                /**
                   Returns a shared instance of this object.

                   The instance() method will always return the same
                   address, though it is potentially possible
                   (post-main()) that the actual object living at that
                   address is different from previous calls.

                   It is never a good idea to hold on to the returned
                   reference for the life of an object, as that bypasses
                   the phoenixing capabilities.

                   If you ever delete it you're on you're own. That's
                   a Bad Idea.
                 */
                static base_type & instance()
                {
                        static this_type meyers;
                        static bool donethat = false;
                        if( this_type::m_destroyed )
                        {
                                phoenix_CERR << "Phoenixing!" << std::endl;
                                donethat = false;
                                new( &meyers ) this_type;
                                atexit( this_type::do_atexit );
                        }
                        if( !donethat )
                        {
                                phoenix_CERR << "initializing instance" << std::endl;
                                donethat = true;
                                initializer_type()( meyers );
                        }
                        phoenix_CERR << "instance() == " <<std::hex<<&meyers<<std::endl;
                        return meyers;
                }

        private:

                /** A convenience typedef. */
                typedef phoenix<base_type,context_type,initializer_type> this_type;

                static bool m_destroyed;

                phoenix()
                {
                        phoenix_CERR << "phoenix() @" << std::hex<< this << std::endl;
                        m_destroyed = false;
                }

                virtual ~phoenix() throw()
                {
                        phoenix_CERR << "~phoenix() @" << std::hex<< this << std::endl;
                        m_destroyed = true;
                }
                /**
                   Destroys the shared object via a manual call to it's dtor.
                 */
                static void do_atexit()
                {
                        if( m_destroyed ) return;
                        phoenix_CERR << "::do_atexit() @ " << std::hex << &instance() << std::endl;
                        static_cast<this_type &>(instance()).~phoenix(); // will eventually trigger the BaseType dtor
                }
        
        };
        template <typename T, typename C, typename I> bool phoenix<T,C,I>::m_destroyed = false;

} } // namespace

#undef phoenix_DEBUG
#undef phoenix_CERR
#endif // s11n_net_s11n_PHOENIX_HPP_INCLUDED
