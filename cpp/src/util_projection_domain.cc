#include "geometry/util_projection_domain.h"

namespace geometry {

namespace util_projection_domain {


void polylineToBoostPolygon(EigenPolyline& polyline, polygon_type& boost_polygon) {
    boost_polygon.clear();
    for (EigenPolyline::iterator it = polyline.begin(); it != polyline.end(); it++) {
        boost::geometry::append(boost_polygon, point_type((*it)[0], (*it)[1]));
    }
}


void polylineToBoostPolygon(const EigenPolyline& polyline, polygon_type& boost_polygon) {
    boost_polygon.clear();
    for (const auto &vert : polyline) {
        double x = vert[0];
        double y = vert[1];
        boost::geometry::append(boost_polygon, point_type(vert[0], vert[1]));
    }
}


void overapproximatePolygonAABB(const EigenPolyline& polyline, polygon_type& boost_poly_aabb) {
    boost_poly_aabb.clear();

    // init extremum points of polygon
    double x_min = std::numeric_limits<double>::infinity();
    double x_max = -std::numeric_limits<double>::infinity();
    double y_min = std::numeric_limits<double>::infinity();
    double y_max = -std::numeric_limits<double>::infinity();

    // iterate over vertices of polygon and store min/max
    for (const auto& vert: polyline) {
        double x = vert[0];
        double y = vert[1];
        x_min = x_min > x ? x : x_min;
        y_min = y_min > y ? y : y_min;
        x_max = x_max < x ? x : x_max;
        y_max = y_max < y ? y : y_max;
    }

    // create polygon
    boost::geometry::append(boost_poly_aabb, point_type(x_min, y_min));
    boost::geometry::append(boost_poly_aabb, point_type(x_min, y_max));
    boost::geometry::append(boost_poly_aabb, point_type(x_max, y_max));
    boost::geometry::append(boost_poly_aabb, point_type(x_max, y_min));
    boost::geometry::append(boost_poly_aabb, point_type(x_min, y_min));
}


std::vector<EigenPolyline> polygonWithinPolygonBoost(const polygon_type& polygon_input,
                                                     const polygon_type& polygon_other) {
    // init output: vector of EigenPolylines
    std::vector<EigenPolyline> polygons_within_polygon_other;

    // create deque for parts of polygon_input in polygon_other
    std::deque<polygon_type> parts_in_polygon;

    // intersect with Boost
    boost::geometry::intersection(polygon_input, polygon_other, parts_in_polygon);

    for (polygon_type const &p : parts_in_polygon) {
        EigenPolyline vertices;
        for (auto it = boost::end(boost::geometry::exterior_ring(p)) - 1;
             (it != boost::begin(boost::geometry::exterior_ring(p))); --it) {
            double x = boost::geometry::get<0>(*it);
            double y = boost::geometry::get<1>(*it);
            vertices.emplace_back(x, y);
        }
        polygons_within_polygon_other.push_back(vertices);
    }
    return polygons_within_polygon_other;
}


std::vector<EigenPolyline> polygonWithinPolygonBoost(const EigenPolyline& polygon_input,
                                                     const polygon_type& polygon_other) {
    // create Boost polygon from EigenPolyline
    polygon_type poly_in;
    polylineToBoostPolygon(polygon_input, poly_in);

    // check intersection
    return polygonWithinPolygonBoost(poly_in, polygon_other);
}

} // namespace util_projection_domain

} // namepsace geometry
