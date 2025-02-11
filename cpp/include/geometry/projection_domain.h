#ifndef PROJECTION_DOMAIN_H
#define PROJECTION_DOMAIN_H

#include <chrono>
#include <vector>

#include "geometry/segment.h"
#include "geometry/util.h"
#include "geometry/util_sweep_line.h"
#include "geometry/util_projection_domain.h"
#include "geometry/clcs_exceptions.h"
#include "geometry/clcs_types.h"
#include "geometry/clcs_logger.h"

// namespaces
namespace util_sweep = geometry::sweep_line_util;
namespace util_proj = geometry::util_projection_domain;

namespace geometry {

class ProjectionDomain {
public:
    /**
     * Class for handling the unique projection domain for a given reference path.
     * Automatically computes both the unique Cartesian and the Curvilinear projection domain.
     * The absolute value of the lateral distance of the projection domain border
     * from the reference path is limited to default_projection_domain_limit.
     *
     * @param seg_list Reference to list of polyline segments of the reference path
     * @param curv_vec Curvature vector of discrete reference path
     * @param curv_rad_vec Curvature radius vector of discrete reference path
     * @param seg_lon_coord_vec Longitudinal coordinate vector of the segment list
     * @param def_proj_domain_lim maximum lateral extent of projection domain
     * @param eps numerical precision value to underapproximate projection domain
     * @param method valid options:
     *               (1) worst-case curvature approximation (very conservative)
     *               (2) tight under-approximation using sweep line (less conservative)
     */
    ProjectionDomain(const std::vector<std::unique_ptr<Segment>> &seg_list,
                     const std::vector<double> curv_vec,
                     const std::vector<double> curv_rad_vec,
                     const std::vector<double> seg_lon_coord_vec,
                     const double length,
                     const double def_proj_domain_lim = 20.0,
                     const double eps = 0.1,
                     const int method = 1);

    /// default destructor
    ~ProjectionDomain() = default;

    /**
     * Getter for the border of the unique projection domain in Cartesian coordinates
     *
     * @return 2D polyline representing the border
     */
    EigenPolyline projectionDomainBorder() const;

    /**
     * Getter for the border of the unique projection domain in curvilinear coordinates
     *
     * @return 2D polyline representing the border
     */
    EigenPolyline curvilinearProjectionDomainBorder() const;

    /**
     * Getter function for upper projection domain border in Cartesian coordinates
     */
    EigenPolyline upperProjectionDomainBorder() const;

    /**
     * Getter function for lower projection domain border in Cartesian coordinates
     */
    EigenPolyline lowerProjectionDomainBorder() const;

    /**
     * Checks if a Cartesian point is within the unique Cartesian projection
     *
     * @param x x-coordinate in the Cartesian coordinate system
     * @param y y-coordinate in the Cartesian coordinate system
     * @return true, if the point is inside or on the boundary of the projection domain
     */
    bool cartesianPointInProjectionDomain(double x, double y) const;

    /**
     * Checks if a curvilinear point is within the unique curvilinear projection
     *
     * @param segment_ptr pointer to segment
     * @param seg_idx ID of segment
     * @param seg_lon_coord longitudinal coordinate of segment start
     * @param s longitudinal coordinate in the curvilinear coordinate system
     * @param l lateral coordinate in the curvilinear coordinate system
     * @return <true,true>, if the point is within the lon. and lat. bounds of the curvilinear projection domain
     */
    std::tuple<bool, bool> curvilinearPointInProjectionDomain(const std::unique_ptr<Segment>& segment_ptr,
                                                              int seg_idx,
                                                              const double seg_lon_coord,
                                                              double s, double l) const;

    /**
     * Computes the parts of a polygon which are inside the Cartesian or Curvilinear unique projection
     *
     * @param polygon vertices of the polygon; vertices must be sorted clockwise and given as closed list
     * @param projection_domain curvilinear or Cartesian projection domain
     * @return parts of the polygon which are inside the projection domain.
     */
    // TODO static method, move to separate utils file
    std::vector<EigenPolyline> polygonWithinProjectionDomain(const EigenPolyline &polygon,
                                                             const polygon_type &projection_domain) const;

    /**
     * Computes the parts of a polygon which are inside the unique projection domain
     *
     * @param polygon vertices of the polygon; vertices must be sorted clockwise and given as closed list
     * @return parts of the polygon which are inside the projection domain.
     */
    std::vector<EigenPolyline> determineSubsetOfPolygonWithinProjectionDomain(const EigenPolyline &polygon) const;

    /**
     * Computes the parts of a polygon (in curvilinear coordinates) which are inside the curvilinear projection domain.
     *
     * @param polygon vertices of the polygon; vertices must be sorted clockwise and given as closed list
     * @return parts of the polygon which are inside the curvilinear projection domain.
     */
    std::vector<EigenPolyline>
    determineSubsetOfPolygonWithinCurvilinearProjectionDomain( const EigenPolyline &polygon) const;

    /**
     * Computes the parts of different multipolygons which are inside the unique projection domain
     * @param[in] polygons list of polygons
     * @param[in] groups_of_polygons list of IDs indicating the group to which a polygon belongs
     * @param[in] num_omp_threads number of OMP threads for computation
     * @param[out] polygons_in_projection_domain polygons within the projection domain
     * @param[out] groups_of_polygons_in_projection_domain list IDs indicating to which group the clipped polygon belongs
     */
    void determineSubsetsOfMultiPolygonsWithinProjectionDomain(
            const std::vector<EigenPolyline> &polygons,
            const std::vector<int> groups_of_polygons, const int num_omp_threads,
            std::vector<EigenPolyline> &polygons_in_projection_domain,
            std::vector<int> &groups_of_polygons_in_projection_domain) const;

private:
    /**
     * Approximates the unique Cartesian projection domain of the coordinate system.
     *
     * @param segment_list: reference to segment list of discrete reference path
     */
    void approximateProjectionDomain(const std::vector<std::unique_ptr<Segment>>& segment_list);

    /**
     * Computes the maximum positive (left) and negative (right) distance from the reference path
     * This function computes the max and min distances based on the worst-case (i.e., highest) curvature
     *
     * @param segment_list: reference to segment list of discrete reference path
     * @return minimum (negative, right) and maximum (positive, left) distance
     */
    std::tuple<double, double> computeProjectionDomainLimits(
            const std::vector<std::unique_ptr<Segment>>& segment_list) const;

    /**
     * Computes the border of the unique projection domain. The points are sorted
     * clockwise and the first point coincides with the last point.
     *
     * @param segment_list: reference to segment list of discrete reference path
     * @param vec_leftSegmentDistances distance for each segment to the left of the reference (positive sign)
     * @param vec_rightSegmentDistances distance for each segment to the right of the reference (negative sign)
     * @return border of projection domain as a polygon
     */
    EigenPolyline computeProjectionDomainBorder(const std::vector<std::unique_ptr<Segment>>& segment_list,
                                                const std::vector<double>& vec_leftSegmentDistances,
                                                const std::vector<double>& vec_rightSegmentDistances);

    /**
     * Computes the border of the unique projection domain. This method utilizes a sweep  line approach to detect the
     * intersection of segment normals. We approximate the evolute of the reference path as the projection domain. This
     * method is less conservative than the computation using computeProjectionDomainBorder(), which limits the projection
     * domain to the minimum radius of the osculating circle.
     *
     * @param [in] segment_list: reference to segment list of discrete reference path
     * @param [out] vec_leftSegmentDistances: distance for each segment to the left of the reference
     * @param [out] vec_rightSegmentDistances: distance for each segment to the right of the reference
     */
    void computeProjectionDomainExact(const std::vector<std::unique_ptr<Segment>>& segment_list,
                                      std::vector<double>& vec_leftSegmentDistances,
                                      std::vector<double>& vec_rightSegmentDistances);

    /**
     * Get the lateral distance of intersection for a given segment. If this segment has intersections with multiple
     * segments, the closest distance ist returned.
     *
     * @param segment_list: reference to segment list of discrete reference path
     * @param vec_segment_distances vector with signed distance for each segment
     * @param map_intersections Maps segment ID to pair of intersecting segment and intersection point
     * @param direction string (possible values: LEFT, RIGHT) indicating the side for which signed distance is computed
     */

    void getSegmentDistances(const std::vector<std::unique_ptr<Segment>>& segment_list,
                             std::vector<double>& vec_segment_distances,
                             std::unordered_map<int, std::vector<std::pair<int, Eigen::Vector2d>>>& map_intersections,
                             std::string direction);

    /**
     * Approximates the curvilinear projection domain of the coordinate system.
     */
    void approximateCurvilinearProjectionDomain(const std::vector<std::unique_ptr<Segment>>& segment_list,
                                                const std::vector<double>& vec_leftSegmentDistances,
                                                const std::vector<double>& vec_rightSegmentDistances);

    /// total length of the reference path (max. longitudinal coordinate)
    const double length_;
    /// reduction of lateral projection domain distance by precision value eps_
    const double eps_;
    /// default maximum lateral extent of projection domain
    const double default_projection_domain_limit_;
    /// method for projection domain computation (valid numbers: 1 or 2)
    const int proj_domain_method_;

    /// curvature vector of discrete reference path
    const std::vector<double> curvature_vec_;
    /// curvature radius vector of discrete reference path
    const std::vector<double> curvature_radius_vec_;
    /// longitudinal coordinates of segments
    const std::vector<double> segment_lon_coord_;

    /// Cartesian projection domain as Boost Polygon
    polygon_type projection_domain_;
    /// Curvilinear projection domain as Boost Polygon
    polygon_type curvilinear_projection_domain_;

    /// Cartesian projection domain border as a polyline
    EigenPolyline projection_domain_border_;
    /// upper (left w.r.t. ref path) Cartesian projection domain border as a polyline
    EigenPolyline upper_projection_domain_border_;
    /// lower (right w.r.t. ref path) Cartesian projection domain border as a polyline
    EigenPolyline lower_projection_domain_border_;
    /// Curvilinear projection domain border as polyline
    EigenPolyline curvilinear_projection_domain_border_;
};

} // namespace geometry


#endif //PROJECTION_DOMAIN_H

