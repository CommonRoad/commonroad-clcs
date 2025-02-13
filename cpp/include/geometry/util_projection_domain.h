#ifndef UTIL_PROJECTION_DOMAIN_H
#define UTIL_PROJECTION_DOMAIN_H

#include <vector>
#include <utility>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include "geometry/clcs_types.h"
#include "geometry/util.h"

namespace geometry {

namespace util_projection_domain {

    /**
     * Converts an EigenPolyline representing the vertices of a polygon to a Boost polygon type
     *
     * @param [in] polyline vertices of a polygon as a closed polyline
     * @param [out] boost_polygon boost polygon which is created from the input polyline
     */
    void polylineToBoostPolygon(EigenPolyline& polyline, polygon_type& boost_polygon);

    /**
     * Converts an EigenPolyline representing the vertices of a polygon to a Boost polygon type
     *
     * @overload
     */
    void polylineToBoostPolygon(const EigenPolyline& polyline, polygon_type& boost_polygon);


    /**
     * Over-approximates a polygon (given by its vertices) with an axis aligned rectangle Boost polygon
     *
     * @param [in] polyline vertices of the polygon as a closed (Eigen) polyline
     * @param [out] boost_poly_aabb AABB over-approximation as boost polygon
     */
    void overapproximatePolygonAABB(const EigenPolyline& polyline, polygon_type& boost_poly_aabb);


    /**
     * Checks whether a polygon (polygon_input) is contained within another polygon (polygon_other).
     * Additionally returns the subset of polygon_input which intersects with the Boost polygon
     * The returned polygon is returned as a polyline describing its vertices
     *
     * @param polygon_input input  polygon (as Boost polygon)
     * @param polygon_other other polygon (as Boost polygon)
     * @return subset of polygon_input which is contained in polygon_other (as EigenPolyline)
     */
    std::vector<EigenPolyline> polygonWithinPolygonBoost(const polygon_type& polygon_input,
                                                         const polygon_type& polygon_other);

    /**
     * Checks whether a polygon (polygon_input) is contained within another polygon (polygon_other).
     *
     * @overload
     */
    std::vector<EigenPolyline> polygonWithinPolygonBoost(const EigenPolyline& polygon_input,
                                                         const polygon_type& polygon_other);

} // namespace util_projection_domain

} // namespace geometry

#endif //UTIL_PROJECTION_DOMAIN_H
