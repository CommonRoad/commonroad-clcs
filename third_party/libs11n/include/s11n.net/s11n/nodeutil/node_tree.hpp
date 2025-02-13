#ifndef s11n_util_NODE_UTIL_HPP_INCLUDED
#define s11n_util_NODE_UTIL_HPP_INCLUDED 1

#include <map>
#include <list>
#include <s11n.net/s11n/traits.hpp>

namespace s11n {

/**
   The s11n::util namespace encapsulates some utility code
   for client-side use alongside the s11n framework.
*/
namespace util {

        /**
           node_tree assist in manipulating child/parent relationships
           in a tree of data nodes.

           NodeT must usable via s11n::node_traits&lt;NodeT&gt;.

	   An example use for this class: the s11nbrowser app
	   uses it to assist in the mapping of s-nodes to
	   UI elements and for cutting/pasting subtrees of nodes.
        */
        template <typename NodeT>
        class node_tree
        {
        public:
                typedef NodeT node_type;
                typedef ::s11n::node_traits< node_type > traits_type;

                /**
                   Ownership of node does not change, and root should
                   outlive this object, or at least live until
                   clime_tree() is called with another root.

                   root may be 0.
                */
                explicit node_tree( node_type * root ) : m_root(root), m_count(0)
                {
                        this->climb_tree( root );
                }

                node_tree() : m_root(0), m_count(0)
                {}

                ~node_tree() {}


                typedef typename traits_type::child_list_type list_type;


                /**
                   Maps relations of parens to children in node tree,
                   using the given root node. Returns the cumulative
                   number of nodes traversed.

                   Ownership of node does not change, so node must
                   outlive this object or, if it is destroyed first,
                   climb_tree() must be called with another node
                   before any other API of this class is called.

                   If node == 0 then this object's data is cleared and
                   0 is returned.

                   The second parameter is for internal use: never
                   pass a value, or pass 0 if you must pass something.
                 */
                size_t climb_tree( node_type * node, size_t internal_use = 0 )
                {
                        if( ! node ) 
			{
				this->clear();
				return 0;
			}
                        else if( 0 == internal_use ) // first iteration: clear state
                        {
                                this->clear();
                                this->m_root = node;
                                this->m_rmap[node] = 0;
                        }
                        ++this->m_count;
                        list_type & chlist = traits_type::children(*node);
                        typename list_type::iterator it = chlist.begin(),
                                et = chlist.end();
                        for( ; et != it; ++it )
                        {
                                this->m_rmap[(*it)] = node;
                                this->climb_tree( (*it), 1 + internal_use );
                        }
                        return this->m_count;
                }

                /**
                   Returns the parent node of node, of 0 if no parent
                   was mapped by climb_tree().

                   The caller does not own the pointer: it is owned by
                   its own parent. If it has no parent, then "someone
                   else" is assumed to have ownership of it (whoever
                   gave it to this object, presumably).
                */
                node_type * parent_of( node_type & node )
                {
                        typename reverse_map_type::iterator it = this->m_rmap.find( &node );
                        if( this->m_rmap.end() == it ) return 0;
                        return (*it).second;
                }

                /**
                   A convenience wrapper around
                   traits_type::children(node).
                */
                list_type & children_of( node_type & node )
                {
                        return traits_type::children(node);
                }

                /**
                   Removes the given node from the tree. If it returns true,
                   the caller owns node, otherwise ownership is unchanged.

                */
                bool take_node( node_type * node )
                {
                        typename reverse_map_type::iterator rit = this->m_rmap.find( node );
                        if( this->m_rmap.end() == rit ) return false;
                        node_type * p = (*rit).second;
                        if( p )
                        {
                                // take node from parent's list...
                                typename list_type::iterator it = traits_type::children(*p).begin(),
                                        et = traits_type::children(*p).end();
                                for( ; et != it; ++it )
                                {
                                        if( (*it) == node )
                                        {
                                                traits_type::children(*p).erase( it );
                                                break;
                                        }
                                }
                        }

                        this->m_rmap.erase(rit);
                        if( node == this->root() )
                        {
				this->clear();
                        }
                        return true;
                }


                /**
                   Reparents node n to parent node newp.

                   Both n and newp are assumed to be under root().

                   n may not be root() and neither n nor newp may be
                   0.

                   Returns true on success, false on error.

                   If this function returns false then ownership is
                   unchanged and there are no side effects. If it
                   returns true, ownership of n *typically* moves to
                   newp. That is, newp will delete n when newp is
                   destroyed. More specifically, that depends on
                   node_type's implementation, but s11n::data_node and
                   s11n::s11n_node both work that way.
                */
                bool reparent( node_type * n, node_type * newp )
                {
                        if( (!n) || (!newp) || (!this->root()) || (this->root() == n) )
                        {
                                return false;
                        }
                        if( ! this->take_child( n ) )
                        {
                                return false;
                        }
                        this->m_rmap[n] = newp;
                        traits_type::children(*newp).push_back( n );
                        return true;
                }

                /**
                   Returns this object's root node. Ownership is not
                   changed.
                */
                node_type * root() { return this->m_root; }

                /**
                   Clears this object's data and sets its root() node
                   to 0. This object does not own root() and therefor
                   does not delete it.
                */
                void clear()
                {
			this->m_count = 0;
                        this->m_rmap.clear();
                        this->m_root = 0;
                }

        private:
                typedef std::map< node_type *, node_type * > reverse_map_type; // children to parens
                reverse_map_type m_rmap;
                node_type * m_root;
                size_t m_count;
        };

} } // namespace util


#endif // s11n_util_NODE_UTIL_HPP_INCLUDED
