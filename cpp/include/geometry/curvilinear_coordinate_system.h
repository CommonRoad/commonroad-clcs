#ifndef CURVILINEAR_COORDINATE_SYSTEM_H
#define CURVILINEAR_COORDINATE_SYSTEM_H

#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <limits>
#include <list>
#include <memory>
#include <numeric>
#include <vector>
#include <optional>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include "geometry/application_settings.h"

#include "geometry/serialize/icurvilinear_coordinate_system_export.h"

#include "geometry/projection_domain.h"
#include "geometry/segment.h"
#include "geometry/path_segments.h"
#include "geometry/util.h"
#include "geometry/util_sweep_line.h"
#include "geometry/clcs_exceptions.h"
#include "geometry/clcs_types.h"
#include "geometry/clcs_logger.h"

namespace util_sweep = geometry::sweep_line_util;

namespace geometry {

class CurvilinearCoordinateSystem;

typedef std::shared_ptr<CurvilinearCoordinateSystem>
    CurvilinearCoordinateSystemPtr;
typedef std::shared_ptr<const CurvilinearCoordinateSystem>
    CurvilinearCoordinateSystemConstPtr;
}  // namespace geometry

namespace geometry {


class CurvilinearCoordinateSystem
    : public std::enable_shared_from_this<CurvilinearCoordinateSystem> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * Creates a curvilinear coordinate system aligned to the given reference
   * path. The unique projection domain along the reference path is
   * automatically computed. The absolute value of the lateral distance of the
   * projection domain border from the reference path is limited to
   * default_projection_domain_limit. To account for numeric imprecisions, the
   * parameter eps reduces the computed lateral distance of the projection
   * domain border from the reference path.
   *
   * @param reference_path 2D polyline in Cartesian coordinates
   * @param default_projection_domain_limit maximum absolute distance of the
   * projection domain border from the reference path
   * @param eps reduces the lateral distance of the projection domain border
   * from the reference path
   * @param eps2 add three additional segments to the beginning the reference
   * path and two segments to the end, with length eps2, to enable the
   * conversion of points near the beginning and the end of the reference path
   * @param log_level: string indicating the logging level for debugging purposes (default off)
   * Valid options: "off", "critical", "err", "warn", "info", "debug", "trace"
   * @param method [int] specify method for computing the projection domain (valid inputs: 1, 2)
   */
  CurvilinearCoordinateSystem(const EigenPolyline& reference_path,
                              double default_projection_domain_limit = 20.,
                              double eps = 0.1,
                              double eps2 = 1e-4,
                              const std::string &log_level = "off",
                              int method = 1);

  /**
   * Getter for default projection domain limit
   */
  double defaultProjectionDomainLimit() const { return default_projection_domain_limit_; }

  /**
   * Getter for eps_ value
   */
  double eps() const { return eps_; }

  /**
   * Getter for eps2_ value
   */
  double eps2() const { return eps2_; }

  /**
   * Getter for method of computing projection domain
   */
  int method() const { return method_; }

  /**
   * Getter for the reference path in Cartesian coordinates.
   * Reference path is extended at start and end (see eps2_)
   *
   * @return reference path as a 2d polyline
   */
  EigenPolyline referencePath() const;

  /**
   * Getter for the original unextended reference path as a 2D polyline
   *
   * @return original reference path as 2D polyline
   */
  EigenPolyline referencePathOriginal() const;

  /**
   * Getter for the partitions of the reference path according to inflection points.
   *
   * @return vector of reference path partitions
   */
  std::vector<EigenPolyline> referencePathPartitions() const;

  /**
   * Getter for the border of the unique projection domain in Cartesian coordinates
   *
   * @return 2D polyline representing the border of the projection domain
   */
  EigenPolyline projectionDomainBorder() const;

  /**
   * Getter for the border of the unique projection domain in curvilinear coordinates
   *
   * @return 2D line string representing the border of the projection domain
   */
  EigenPolyline curvilinearProjectionDomainBorder() const;

  /**
   * Getter for upper projection domain border
   *
   * @return 2D polyline representing upper border
   */
  EigenPolyline upperProjectionDomainBorder() const;

  /**
   * Getter for lower projection domain border
   *
   * @return 2D polyline representing lower border
   */
  EigenPolyline lowerProjectionDomainBorder() const;

  /**
   * Getter for the length of the reference path
   *
   * @return total length
   */
  double length() const;

  /**
   * Getter for curvature vector.
   */
  std::vector<double> curvatureVector() const;

  /**
   * Getter for curvature radius vector.
   */
  std::vector<double> curvatureRadiusVector() const;

  /**
   * Getter for the maximum curvature radius along the reference path .
   *
   * @return maximum curvature radius
   */
  double maximumCurvatureRadius() const;

  /**
   * Getter for the minimum curvature radius along the reference path.
   *
   * @return minimum curvature radius
   */
  double minimumCurvatureRadius() const;

  /**
   * Getter for the maximum curvature along the reference path
   *
   * @return maximum curvature
   */
  double maximumCurvature() const;

  /**
   * Getter for the minimum curvature along the reference path
   *
   * @return minimum curvature
   */
  double minimumCurvature() const;

  /**
   * Getter returning reference to vector of the longitudinal coordinates of the path segments
   *
   * @return reference to vector with s coordinates for each segment
   */
  const std::vector<double> &segmentsLongitudinalCoordinates() const;

  /**
   * Getter for returning a reference to the list of path segments
   *
   * @return reference to vector segment pointers
   */
  const std::vector<std::unique_ptr<Segment>> &getSegmentList() const;

  /**
   * Setter for logging level
   *
   * @param log_level desired logging level given as a string
   * Valid options: "off", "critical", "err", "warn", "info", "debug", "trace"
   */
  void setLoggingLevel(const std::string &log_level);

  /**
   * Setter for the curvature vector of the reference path
   *
   * @param curvature of the reference path
   */
  void setCurvature(const std::vector<double> curvature);

  /**
   * Computes and sets the curvature vector for the reference path.
   *
   * @param digits: no. of decimal points for curvature value (default 8)
   */
  int computeAndSetCurvature(int digits = 8);

  /**
   * Returns an interval of the curvatures within a given range of longitudinal positions.
   *
   * @param s_min minimum longitudinal position
   * @param s_max maximum longitudinal position
   * @return enclosing interval of curvature values of the reference path within [s_min, s_max]
   */
  std::tuple<double, double> curvatureRange(double s_min, double s_max) const;

  /**
   * Normal vector at a specific longitudinal coordinate
   *
   * @param s longitudinal coordinate
   * @return normal vector
   */
  Eigen::Vector2d normal(double s) const;

  /**
   * Tangent vector at a specific longitudinal coordinate
   *
   * @param s longitudinal coordinate
   * @return tangent vector
   */
  Eigen::Vector2d tangent(double s) const;

  /**
   * Transforms a curvilinear point (s, l) to Cartesian coordinates
   *
   * @param s longitudinal coordinate
   * @param l lateral coordinate
   * @param check_proj_domain If True, it is checked before transformation whether a point is within the projection
   * domain. If the check fails, an error is thrown.
   * @return point in global coordinates
   */
  Eigen::Vector2d convertToCartesianCoords(double s, double l, bool check_proj_domain = true) const;

  /**
   * Transforms a Cartesian point (x, y) to curvilinear coordinates.
   *
   * @param x x-coordinate in the Cartesian coordinate system
   * @param y y-coordinate in the Cartesian coordinate system
   * @param check_proj_domain If True, it is checked before transformation whether a point is within the projection
   * domain. If the check fails, an error is thrown.
   * @return point in the curvilinear frame.
   */
  Eigen::Vector2d convertToCurvilinearCoords(double x, double y, bool check_proj_domain = true) const;

  /**
   * Transforms a Cartesian point (x, y) to curvilinear coords and returns the corresponding segment index.
   *
   * @param[in] x x-coordinate in the Cartesian coordinate system
   * @param[in] y y-coordinate in the Cartesian coordinate system
   * @param[out] segment index, in which the point is contained
   * @param check_proj_domain If True, it is checked before transformation whether a point is within the projection
   * domain. If the check fails, an error is thrown.
   * @return point in the curvilinear frame.
   */
  Eigen::Vector2d convertToCurvilinearCoordsAndGetSegmentIdx(
          double x, double y, int &segment_idx, bool check_proj_domain = true) const;

  /**
   * Transforms a rectangle in curvilinear coordinates to the Cartesian coordinates.
   * Additionally, a triangle mesh of the resulting polygon is generated.
   *
   * @param[in] s_lo minimum longitudinal coordinate of the rectangle
   * @param[in] s_hi maximum longitudinal coordinate of the rectangle
   * @param[in] l_lo minimum lateral coordinate of the rectangle.
   * @param[in] l_hi maximum lateral coordinate of the rectangle
   * @param[out] triangle_mesh
   * @return transformed rectangle in Cartesian coordinates
   */
  EigenPolyline convertRectangleToCartesianCoords(double s_lo, double s_hi, double l_lo, double l_hi,
                                                  std::vector<EigenPolyline> &triangle_mesh) const;

  /**
   * Converts list of Cartesian points to curvilinear points.
   *
   * @param points vector of points in the global coordinate frame
   * @param num_omp_threads number of OMP threads for computation
   * @return transformed points
   */
  EigenPolyline convertListOfPointsToCurvilinearCoords(const EigenPolyline &points, int num_omp_threads) const;

  /**
   * Converts list of curvilinear points to Cartesian points
   *
   * @param points vector of points in the curvilinear coordinate system
   * @param num_omp_threads number of OMP threads for computation
   * @return transformed points
   */
  EigenPolyline convertListOfPointsToCartesianCoords(const EigenPolyline &points, int num_omp_threads) const;


  /**
   * Transforms a polygon to the curvilinear coordinate system.
   *
   * @param[in] polygon input polygon
   * @param[out] transformed_polygon transformed polygon
   */
  void convertPolygonToCurvilinearCoords(const EigenPolyline &polygon,
                                         std::vector<EigenPolyline> &transformed_polygon) const;

  /**
   * Transforms different multipolygons from Cartesian to curvilinear coords. Furthermore, the transformed
   * multipolygons are rasterized, i.e., they are over-approximated with axis-aligned rectangles.
   *
   * @param[in] polygons list of polygons
   * @param[in] groups_of_polygons list of IDs indicating the group to which a polygon belongs
   * @param[in] num_polygon_groups number of different polygon groups
   * @param[in] num_omp_threads number of OMP threads for computation
   * @param[out] transformed_polygons transformed polygons
   * @param[out] transformed_polygons_rasterized transformed and rasterized polygons
   */
  void convertListOfPolygonsToCurvilinearCoordsAndRasterize(
      const std::vector<EigenPolyline> &polygons,
      const std::vector<int> groups_of_polygons, int num_polygon_groups,
      int num_omp_threads,
      std::vector<std::vector<EigenPolyline>> &transformed_polygons,
      std::vector<std::vector<EigenPolyline>> &transformed_polygons_rasterized)
      const;

  /**
   * Validates if a Cartesian point (x, y) is within the unique Cartesian projection.
   *
   * @param x x-coordinate in the Cartesian coordinate system
   * @param y y-coordinate in the Cartesian coordinate system
   * @return true, if the point is inside or on the boundary of the projection
   domain, false, if the point is outside of the boundary of the projection
   domain.
   */
  bool cartesianPointInProjectionDomain(double x, double y) const;

  /**
   * Validates if a curvilinear point (s, l) is within the unique curvilinear projection.
   *
   * @param seg_idx segment index (if determined before already)
   * @param s longitudinal coordinate in the curvilinear coordinate system
   * @param l lateral coordinate in the curvilinear coordinate system
   * @return <true,true>, if the point is within the longitudinal and lateral bounds of the projection domain
    */
  std::tuple<bool, bool> curvilinearPointInProjectionDomain(int seg_idx, double s, double l) const;

  /**
   * Validates if a curvilinear point (s, l) is within the unique curvilinear projection.
   *
   * @overload
   */
  std::tuple<bool, bool> curvilinearPointInProjectionDomain(double s, double l) const;

  /**
   * Computes the parts of a polygon which are inside the unique Cartesian projection domain.
   *
   * @param polygon vertices of the boundary of the polygon; vertices must be sorted clockwise and given as closed list
   * @return parts of the polygon which are inside the projection domain.
   */
  std::vector<EigenPolyline> determineSubsetOfPolygonWithinProjectionDomain(const EigenPolyline &polygon) const;

  /**
   * Computes the parts of a polygon which are inside the unique curvilinear projection domain.
   *
   * @param polygon vertices of the boundary of the polygon.
   * @return parts of the polygon which are inside the curvilinear projection domain.
   */
  std::vector<EigenPolyline>
  determineSubsetOfPolygonWithinCurvilinearProjectionDomain(const EigenPolyline &polygon) const;


  /**
   * Computes the parts of different multipolygons which are inside the projection domain.
   *
   * @param[in] polygons list of polygons
   * @param[in] groups_of_polygons list of IDs indicating the group to which a polygon belongs
   * @param[in] num_omp_threads number of OMP threads for computation
   * @param[out] polygons_in_projection_domain polygons within the projection domain
   * @param[out] groups_of_polygons_in_projection_domain indices indicating to
   * which group the clipped polygon belongs
   */
  void determineSubsetsOfMultiPolygonsWithinProjectionDomain(
      const std::vector<EigenPolyline> &polygons,
      const std::vector<int> groups_of_polygons, const int num_omp_threads,
      std::vector<EigenPolyline> &polygons_in_projection_domain,
      std::vector<int> &groups_of_polygons_in_projection_domain) const;

#if ENABLE_SERIALIZER
  serialize::ICurvilinearCoordinateSystemExport *exportThis(void) const;

  int serialize(std::ostream &output_stream) const;
  static CurvilinearCoordinateSystemConstPtr deserialize(
      std::istream &output_stream);

#endif

private:
  /**
   * Converts groups of points to the curvilinear coordinate system.
   *
   * @param groups_of_points vector of Cartesian points
   * @param num_omp_threads number of OMP threads for computation
   * @return groups of transformed points and their corresponding segment indices
   */
  std::vector<std::vector<std::tuple<int, double, double>>>
  convertToCurvilinearCoords(const std::vector<EigenPolyline> &groups_of_points,
                             int num_omp_threads) const;

  /**
   * The function computes the corresponding curvilinear points of a polygon for intermediate segments.
   *
   * @param polygon polygon boundary in global coordinates
   * @param curvilinear_coordinates_and_segment_idx transformed boundary of
   * polygons and corresponding segment indices
   * @param curvilinear_polygon complete transformed boundary of polygon
   * @param segment_indices
   */
  bool addPointsAtSegmentTransition(
      const EigenPolyline &polygon,
      const std::vector<std::tuple<int, double, double>>
          &curvilinear_coordinates_and_segment_idx,
      EigenPolyline &curvilinear_polygon, std::set<int> &segment_indices) const;

  /**
   * Over-approximates polygons in the curvilinear coordinate system with axis-aligned rectangles.
   *
   * @param transformed_polygon polygons in curvilinear coordinates
   * @param segment_indices IDs of all segments containing all transformed
   * polygons
   * @param transformed_polygon_rasterized over-approximated polygons
   */
  void rasterizeListOfTransformedPolygonsInProjectionDomain(
      const std::vector<EigenPolyline> &transformed_polygon,
      const std::set<int> &segment_indices,
      std::vector<EigenPolyline> &transformed_polygon_rasterized) const;

  /**
   * Over-approximates a polygon in the curvilinear coordinate system with axis-aligned rectangles.
   *
   * @param transformed_polygon polygon in curvilinear coordinates
   * @param segment_indices  IDs of all segments containing the transformed
   * polygon
   * @param transformed_polygon_rasterized over-approximated polygon
   */
  void rasterizeTransformedPolygonInProjectionDomain(
      const EigenPolyline &transformed_polygon,
      const std::set<int> &segment_indices,
      std::vector<EigenPolyline> &transformed_polygon_rasterized) const;

  /**
   * Given a set of candidate segments for each point, the transformed
   * longitudinal and lateral coordinates for each candidate segment, the
   * function computes the curvilinear coordinates.
   */
  void determineCurvilinearCoordinatesAndSegmentIdx(
      const std::vector<std::vector<std::tuple<int, int>>>
          &candidate_segments_of_points,
      const std::vector<Eigen::RowVectorXd,
                        Eigen::aligned_allocator<Eigen::RowVectorXd>> &s_coord,
      const std::vector<Eigen::RowVectorXd,
                        Eigen::aligned_allocator<Eigen::RowVectorXd>> &l_coord,
      int num_omp_threads,
      std::vector<std::vector<std::tuple<int, double, double>>>
          &groups_of_curvil_points) const;

  std::unique_ptr<ProjectionDomain> projection_domain_ptr;  ///< pointer to handler of the projection domain

  std::unique_ptr<PathSegments> path_segments_ptr;          ///< pointer to reference path segments of the CLCS

  EigenPolyline reference_path_original_;                   ///< original reference path (without extensions)
  std::vector<EigenPolyline> reference_path_partitions_;    ///< inflection point partitions of reference path

  ///< max/min values for curvature and curvature radius
  double max_curvature_radius_{std::numeric_limits<double>::quiet_NaN()};
  double min_curvature_radius_{std::numeric_limits<double>::quiet_NaN()};
  double max_curvature_{std::numeric_limits<double>::quiet_NaN()};
  double min_curvature_{std::numeric_limits<double>::quiet_NaN()};

  std::vector<double> curvature_vec_;           ///< vector of curvatures of reference path
  std::vector<double> curvature_radius_vec_;    ///< vector of curvature radius of reference path
  std::map<double, double> curvature_;          ///< curvature value at longitudinal positions of reference path

  double default_projection_domain_limit_;      ///< default lateral projection domain limit
  double eps_;                                  ///< precision value to reduce lateral limit
  double eps2_;                                 ///< extrapolation value for start and end of reference
  int method_;                                  ///< method for computing the projection domain (valid inputs: 1, 2)

};

}  // namespace geometry

#endif  // CURVILINEAR_COORDINATE_SYSTEM_H
