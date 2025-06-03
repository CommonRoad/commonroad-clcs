#include "pybind.h"

#include "geometry/curvilinear_coordinate_system.h"
#include "geometry/segment.h"
#include "geometry/util.h"

#include <nanobind/eigen/dense.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>
#include <vector>
#include <string>

namespace nb = nanobind;
using namespace nb::literals;

using RowMatrixXd = geometry::RowMatrixXd;

NB_MODULE(pycrccosy, module) {
  export_geometry(module);

  auto mutil_geom = module.def_submodule(
      "Util",
      "Util is a submodule of pycrccosy containing auxiliary functions");

  mutil_geom.def("resample_polyline",
                 [](const Eigen::Ref<const RowMatrixXd> &polyline, double step) {
                   RowMatrixXd ret;
                   geometry::util::resample_polyline(polyline, step, ret);
                   return ret;
                 });

  mutil_geom.def("chaikins_corner_cutting",
                 [](const Eigen::Ref<const RowMatrixXd> &polyline, int refinements) {
                   RowMatrixXd ret;
                   geometry::util::chaikins_corner_cutting(polyline, refinements, ret);
                   return ret;
                 });

  mutil_geom.def("lane_riesenfeld_subdivision",
                 [](const Eigen::Ref<const RowMatrixXd> &polyline, int degree, int refinements) {
                    RowMatrixXd ret_polyline;
                    geometry::util::lane_riesenfeld_subdivision(polyline,
                                                                degree,
                                                                refinements,
                                                                ret_polyline);
                    return ret_polyline;
                });


  mutil_geom.def("compute_pathlength",
                 nb::overload_cast<const geometry::EigenPolyline&>(&geometry::util::computePathlength),
                 "polyline"_a);

  mutil_geom.def("compute_curvature",
                 nb::overload_cast<const geometry::EigenPolyline&>(&geometry::util::computeCurvature),
                 "polyline"_a);

  mutil_geom.def("compute_curvature",
                 nb::overload_cast<const geometry::EigenPolyline&, int>(&geometry::util::computeCurvature),
                 "polyline"_a,
                 "digits"_a);

  mutil_geom.def("get_inflection_point_idx",
                 [](const geometry::EigenPolyline &polyline, int digits) {
                    return geometry::util::getInflectionPointsIdx(polyline, digits);
                });

  mutil_geom.def("intersection_segment_segment",
                 [](const Eigen::Vector2d &l1_pt1, const Eigen::Vector2d &l1_pt2,
                    const Eigen::Vector2d &l2_pt1, const Eigen::Vector2d &l2_pt2) {

                   Eigen::Vector2d intersection_point;

                   bool intersect = geometry::util::intersectionSegmentSegment(l1_pt1, l1_pt2,
                                                                               l2_pt1, l2_pt2,
                                                                               intersection_point);

                   return std::make_tuple(intersect, intersect ? std::optional{intersection_point} : std::nullopt);
                 },
                 "Checks if two segments intersect; if yes, returns their intersection point"
                 "\n\n:param l1_pt1: start point of first segment"
                 "\n\n:param l1_pt2: end point of first segment"
                 "\n\n:param l2_pt1: start point of second segment"
                 "\n\n:param l2_pt2: end point of second segment"
                 "\n\n:return: Tuple with boolean for intersection and intersection point");


}

void export_geometry(const nb::module_ &module) {

  // Bind custom exceptions
  // Disable clang-tidy lints that complain about exceptions not being thrown
  //NOLINTBEGIN
  nb::exception<geometry::CurvilinearProjectionDomainLongitudinalError>
          (module, "CurvilinearProjectionDomainLongitudinalError");
  nb::exception<geometry::CurvilinearProjectionDomainLateralError>
          (module, "CurvilinearProjectionDomainLateralError");
  nb::exception<geometry::CartesianProjectionDomainError>
          (module, "CartesianProjectionDomainError");
  //NOLINTEND

  // class Segment()
  nb::class_<geometry::Segment>(module, "Segment")
      .def(nb::init<const Eigen::Vector2d &, Eigen::Vector2d &,
                    Eigen::Vector2d &, Eigen::Vector2d &>(),
           ":param p_1: start point of the segment\n:param p_2: end point of "
           "the segment\n:param t_1: tangent vector at the start of the "
           "segment\n:param t_2: tangent vector at the end of the segment")

      .def_prop_ro("pt_1", &geometry::Segment::pt_1,
                             ":return: start point of segment in Cartesian coordinates")
      .def_prop_ro("pt_2", &geometry::Segment::pt_2,
                             ":return: end point of segment in Cartesian coordinates")
      .def_prop_ro("length", &geometry::Segment::length,
                                ":return: segment length")
      .def_prop_ro("normal_segment_start", &geometry::Segment::normalSegmentStart,
                             ":return: normal vector at the start of the segment")
      .def_prop_ro("normal_segment_end", &geometry::Segment::normalSegmentEnd,
                             ":return: normal vector at the end of the segment")
      .def_prop_ro("tangent_segment_start", &geometry::Segment::tangentSegmentStart,
                             ":return: tangent vector at the start of the segment")
      .def_prop_ro("tangent_segment_end", &geometry::Segment::tangentSegmentEnd,
                             ":return: tangent vector at the end of the segment");

  // class CurvilinearCoordinateSystem()
  nb::class_<geometry::CurvilinearCoordinateSystem>(
      module, "CurvilinearCoordinateSystem")
      .def(nb::init<const geometry::EigenPolyline &, double, double, double,  const std::string &, int>(),
              "reference_path"_a,
              "default_projection_domain_limit"_a = 20.0,
              "eps"_a = 0.1,
              "eps2"_a = 0.01,
              "log_level"_a = "off",
              "method"_a = 1,
          "Creates a curvilinear coordinate system aligned to the given reference path."
    "The unique projection domain along the reference path is automatically computed."
    "The absolute value of the lateral distance of the projection domain border from the reference path is"
    "limited to default_projection_domain_limit."
    "To account for numeric imprecisions, the parameter eps reduces the computed lateral distance of the"
    "projection domain border from the reference path."
    "\n\n:param reference_path: 2D polyline in Cartesian coordinates"
    "\n\n:param default_projection_domain_limit: maximum absolute distance of the projection domain border"
    "from the reference path, defaults to 20"
    "\n\n:param eps: reduces the lateral distance of the projection domain border from the reference path, defaults to 0.1"
    "\n\n:param eps2: if nonzero, add additional segments to the beginning (3 segments) and the end (2 segments) of the"
    "reference path to avoid numerical imprecisions near the start and end of the reference path."
    "\n\n:param log_level: string indicating the logging level for debugging purposes (default off)"
    "\n\n:param method: method for computing the projection domain (valid inputs: 1, 2)"
          )
      .def("length", &geometry::CurvilinearCoordinateSystem::length,
           ":return: length of the reference path of the curvilinear "
           "coordinate system")

      .def("reference_path",
           &geometry::CurvilinearCoordinateSystem::referencePath,
           "Returns the reference path (extended in both directions by "
           "default, please refer to the parameter of the class constructor "
           "eps2 for more details)\n\n:Returns: "
           "2D polyline representing the reference path (extended)")

      .def("reference_path_original",
           &geometry::CurvilinearCoordinateSystem::referencePathOriginal,
           "Returns the initial reference path without its "
           "extension\n\n:Returns: "
           "2D polyline representing the reference path (original)")

       .def("reference_path_partitions",
            &geometry::CurvilinearCoordinateSystem::referencePathPartitions,
            "Returns the partitions of the reference path according to inflection points.\n\n"
            "Returns: Reference path partitions as a list.")

      .def("projection_domain",
          &geometry::CurvilinearCoordinateSystem::projectionDomainBorder,
          "Returns the border of the unique projection domain of the "
          "curvilinear coordinate system in Cartesian coordinates\n\n:Returns: "
          "2D line string representing the border of the projection domain")

      .def("curvilinear_projection_domain",
           &geometry::CurvilinearCoordinateSystem::curvilinearProjectionDomainBorder,
           "Returns the border of the unique projection domain of the "
           "curvilinear coordinate system in curvilinear "
           "coordinates\n\n:Returns: 2D line string representing the border of "
           "the projection domain")

      .def("upper_projection_domain_border",
           &geometry::CurvilinearCoordinateSystem::upperProjectionDomainBorder,
           "Returns the upper border of the projection domain")

      .def("lower_projection_domain_border",
           &geometry::CurvilinearCoordinateSystem::lowerProjectionDomainBorder,
           "Returns the lower border of the  projection domain")

      .def("segments_longitudinal_coordinates",
           &geometry::CurvilinearCoordinateSystem::segmentsLongitudinalCoordinates,
           ":return: array containing longitudinal coordinates corresponding "
           "to the start point of each segment")

      .def("set_logging_level", &geometry::CurvilinearCoordinateSystem::setLoggingLevel,
           "Setter for logging level"
           "\n\n:param log_level desired logging level given as a string")

      .def("set_curvature", &geometry::CurvilinearCoordinateSystem::setCurvature,
          "Currently, the curvature of the reference path is not computed "
          "automatically upon construction. "
          "The function sets the curvature information manually. "
          "Note that the validity of the curvature is not checked, e.g., if it indeed corresponds to the reference path"
		  "of the curvilinear coordinate system. For an automatic version, please also refer to the member function compute_and_set_curvature. \n\n:param curvature: curvature of the reference path")

      .def("get_curvature", &geometry::CurvilinearCoordinateSystem::curvatureVector,
           "Returns the curvature of the reference path as a list."
           "If the returned list is empty, please set the curvature first using"
           "the member function compute_and_set_curvature(). \n\n:Returns: list with curvature of reference path")

      .def("get_curvature_radius", &geometry::CurvilinearCoordinateSystem::curvatureRadiusVector,
           "Returns the curvature radius of the reference path as a list."
           "If the returned list is empty, please set the curvature first using"
           "the member function compute_and_set_curvature(). \n\n:Returns: list with curvature of reference path")

      .def("curvature_range", &geometry::CurvilinearCoordinateSystem::curvatureRange,
           "Returns an interval of the curvatures of the reference path within "
           "a given range of longitudinal positions\n\n:param s_min: minimum "
           "longitudinal position\n:param s_max: maximum longitudinal "
           "position\n\n:return: enclosing interval of curvature values of the "
           "reference path within the range [s_min, s_max]")

      .def("maximum_curvature_radius",
           &geometry::CurvilinearCoordinateSystem::maximumCurvatureRadius,
           "Returns the maximum curvature radius along the reference path of "
           "the curvilinear coordinate system\n\n:Returns: maximum curvature "
           "radius")

      .def("minimum_curvature_radius",
           &geometry::CurvilinearCoordinateSystem::minimumCurvatureRadius,
           "Returns the minimum curvature radius along the reference path of "
           "the curvilinear coordinate system\n\n:Returns: minimum curvature "
           "radius")

      .def("maximum_curvature",
           &geometry::CurvilinearCoordinateSystem::maximumCurvature,
           "Returns the maximum curvature along the reference path of the "
           "curvilinear coordinate system.\n\n:Returns: maximum curvature")

      .def("minimum_curvature",
           &geometry::CurvilinearCoordinateSystem::minimumCurvature,
           "Returns the minimum curvature along the reference path of the "
           "curvilinear coordinate system.\n\n:Returns: minimum curvature")

      .def("get_segment_list",
           [](const geometry::CurvilinearCoordinateSystem &cosy){
               std::vector<geometry::Segment> vec_segment_copy;
               vec_segment_copy.reserve(cosy.getSegmentList().size());
               for (const auto &ptr : cosy.getSegmentList()) {
                   vec_segment_copy.emplace_back(*ptr);
               }
               return vec_segment_copy;
            })

      .def("normal", &geometry::CurvilinearCoordinateSystem::normal,
           "Normal vector at a specific longitudinal coordinate.\n\n:param s: "
           "longitudinal coordinate.\n\n:Returns: normal vector")

      .def("tangent", &geometry::CurvilinearCoordinateSystem::tangent,
           "Tangent vector at a specific longitudinal coordinate.\n\n:param s: "
           "longitudinal coordinate\n\n:Returns: tangent vector")

      .def("convert_to_cartesian_coords",
           &geometry::CurvilinearCoordinateSystem::convertToCartesianCoords,
           "s"_a,
           "l"_a,
           "check_proj_domain"_a = true,
           "Transforms a point in the curvilinear coordinate frame to the "
           "global coordinate frame."
           "\n\n:param s: longitudinal coordinate"
           "\n\n:param l: lateral coordinate"
           "\n\n:param check_proj_domain: If True, it is checked before transformation whether a point is within the"
           "projection domain."
           "\n\n:Returns: point in Cartesian coordinates")

      .def("cartesian_point_inside_projection_domain",
           &geometry::CurvilinearCoordinateSystem::cartesianPointInProjectionDomain,
           "Validates if a point in global coordinates is within the unique "
           "projection domain.\n\n:param x: x-coordinate in the Cartesian coordinate system"
           "\n\n:param y: y-coordinate in the Cartesian coordinate system\n\n:Returns: True or False")

      .def("curvilinear_point_inside_projection_domain",
           [](const geometry::CurvilinearCoordinateSystem &cosy,
              const double s, const double l){
            const auto &[lon, lat] = cosy.curvilinearPointInProjectionDomain(s, l);
            return lon && lat;
            },
           "Validates if a point in global coordinates is within the unique "
           "projection domain.\n\n:param x: x-coordinate in the Cartesian coordinate system"
           "\n\n:param y: y-coordinate in the Cartesian coordinate system\n\n:Returns: True or False")

      .def("determine_subset_of_polygon_within_projection_domain",
           &geometry::CurvilinearCoordinateSystem::determineSubsetOfPolygonWithinProjectionDomain,
           "Computes the parts of a polygon which are inside the unique "
           "projection domain of the curvilinear coordinate system.\n\n:param "
           "polygon: vertices of the boundary of the polygon; vertices must be "
           "sorted clockwise and given as closed list (the last vertex must be "
           "the same as the first one)\n\n:Returns: parts of the polygon which "
           "are inside the projection domain.")

      .def("determine_subsets_of_multi_polygons_within_projection_domain",
           [](const geometry::CurvilinearCoordinateSystem &cosy,
              const std::vector<geometry::EigenPolyline> &polygons,
              const std::vector<int> &groups_of_polygons,
              const int num_omp_threads) {
             std::vector<geometry::EigenPolyline> polygons_in_projection_domain;
             std::vector<int> groups_of_polygons_in_projection_domain;
             cosy.determineSubsetsOfMultiPolygonsWithinProjectionDomain(
                 polygons, groups_of_polygons, num_omp_threads,
                 polygons_in_projection_domain,
                 groups_of_polygons_in_projection_domain);
             return std::make_tuple(polygons_in_projection_domain, groups_of_polygons_in_projection_domain);
           },
           "Intersects each of the input polygons with the projection domain "
           "and returns the result of the intersection.\n\n:param polygons: "
           "input list of polygons\n\n:param groups_of_polygons: list of "
           "integers (of the same length as the list of input polygons) "
           "indicating the integer ID of the polygon group for each input "
           "polygon. It is useful because the list of output polygons after "
           "the intersection may be shorter and we return the polygon group "
           "IDs for each of the output polygons.\n\n:param num_omp_threads: "
           "number of OMP threads for parallel computation\n\n:return: (list "
           "of polygons produced after the intersection, list containing group "
           "ID for each of the output polygons).")

      .def("determine_subset_of_polygon_within_curvilinear_projection_domain",
           &geometry::CurvilinearCoordinateSystem::determineSubsetOfPolygonWithinCurvilinearProjectionDomain,
           "Computes the parts of a polygon (given in curvilinear coordinates) "
           "which are inside the curvilinear projection domain of the "
           "curvilinear coordinate system.\n\n:param polygon: vertices of the "
           "boundary of the polygon; vertices must be sorted clockwise and "
           "given as closed list (the last vertex must be the same as the "
           "first one)\n\n:Returns: parts of the polygon which are inside the "
           "curvilinear projection domain.")

      .def("convert_to_curvilinear_coords",
           nb::overload_cast<double, double, bool>(&geometry::CurvilinearCoordinateSystem::convertToCurvilinearCoords, nb::const_),
           "x"_a,
           "y"_a,
           "check_proj_domain"_a = true,
           "Transforms a Cartesian point to the curvilinear frame."
           "\n\n:param x: x-coordinate in the Cartesian coordinate system"
           "\n\n:param y: y-coordinate in the Cartesian coordinate system"
           "\n\n:param check_proj_domain: If True, it is checked before transformation whether a point is within the"
           "projection domain."
           "\n\n:return: point in curvilinear coordinates.")

      .def("convert_to_curvilinear_coords_and_get_segment_idx",
           [](const geometry::CurvilinearCoordinateSystem &cosy, double x, double y, bool check_proj_domain) {
             int idx = -1;
             Eigen::Vector2d tmp =
                 cosy.convertToCurvilinearCoordsAndGetSegmentIdx(x, y, idx, check_proj_domain);
             nb::list out;
             out.append(tmp);
             out.append(idx);
             return out;
           },
           "x"_a,
           "y"_a,
           "check_proj_domain"_a = true,
           "Transforms a Cartesian point to the curvilinear frame and returns "
           "the segment index, in which the point is contained."
           "\n\n:param x: x-coordinate in the Cartesian coordinate system"
           "\n\n:param y: y-coordinate in the Cartesian coordinate system"
           "\n\n:return: list [point in curvilinear coordinates, associated segment index]")

      .def("convert_list_of_polygons_to_curvilinear_coords_and_rasterize",
          [](const geometry::CurvilinearCoordinateSystem &cosy,
             const std::vector<geometry::EigenPolyline> &polygons,
             const std::vector<int> &groups_of_polygons, int num_polygon_groups,
             int num_omp_threads) {
            std::vector<std::vector<geometry::EigenPolyline>> transformed_polygon;
            std::vector<std::vector<geometry::EigenPolyline>> transformed_polygon_rasterized;

            cosy.convertListOfPolygonsToCurvilinearCoordsAndRasterize(
                polygons, groups_of_polygons, num_polygon_groups,
                num_omp_threads, transformed_polygon,
                transformed_polygon_rasterized);

            return std::make_tuple(transformed_polygon, transformed_polygon_rasterized);
          },
          "Transforms polygons in the Cartesian coordinates to the curvilinear "
          "coordinates.\n\n:param polygons: list of input polygons\n\n:param "
          "groups_of_polygons: group ID (from 0 to num_polygon_groups-1) for "
          "each input polygon, list of integers\n\n:param num_polygon_groups: "
          "number of polygon groups\n\n:param num_omp_threads: number of OMP "
          "threads for parallel computation\n\n:return: (list of lists of "
          "output polygons, list of lists of rasterized output polygons). Both "
          "outermost lists contain lists of output polygons - for each polygon "
          "group")

      .def("convert_polygon_to_curvilinear_coords",
           [](const geometry::CurvilinearCoordinateSystem &cosy,
              const geometry::EigenPolyline &polygon) {
             // create output polygon
             std::vector<geometry::EigenPolyline> transformed_polygon;

             // call cpp method for converting polygon
             cosy.convertPolygonToCurvilinearCoords(polygon, transformed_polygon);

             // return
             return transformed_polygon;
           },
           "Transforms a polygon to the curvilinear coordinate system.")

      .def("convert_rectangle_to_cartesian_coords",
           [](const geometry::CurvilinearCoordinateSystem &cosy, double s_lo,
              double s_hi, double l_lo, double l_hi) {
             std::vector<geometry::EigenPolyline> triangle_mesh;
             geometry::EigenPolyline transformed_rectangle =
                 cosy.convertRectangleToCartesianCoords(s_lo, s_hi, l_lo, l_hi,
                                                        triangle_mesh);

             return std::make_tuple(transformed_rectangle, triangle_mesh);
           },
           "Transforms a rectangle in the curvilinear coordinates to the "
           "Cartesian coordinates. Additionally, a triangle mesh of the "
           "resulting polygon is generated.\n\n:param s_lo: minimum "
           "longitudinal coordinate of the rectangle\n\n:param s_hi: maximum "
           "longitudinal coordinate of the rectangle\n\n:param l_lo: minimum "
           "lateral coordinate of the rectangle.\n\n:param l_hi: maximum "
           "lateral coordinate of the rectangle\n\n:return: (transformed "
           "rectangle in Cartesian coordinates, triangle_mesh)")

      .def("convert_list_of_points_to_curvilinear_coords",
           &geometry::CurvilinearCoordinateSystem::
               convertListOfPointsToCurvilinearCoords,
           "Converts list of points to the curvilinear coordinate "
           "system.\n\n:param points: vector of points in the global "
           "coordinate frame\n\n:param num_omp_threads: number of OMP threads "
           "for computation\n\n:return: transformed points")

      .def("convert_list_of_points_to_cartesian_coords",
           &geometry::CurvilinearCoordinateSystem::
               convertListOfPointsToCartesianCoords,
           "Converts list of points to the cartesian coordinate system."
           "\n\n:param points: vector of points in the curvilinear coordinate frame"
           "\n\n:param num_omp_threads: number of OMP threads for computation"
           "\n\n:return: transformed points")

      .def("compute_and_set_curvature", &geometry::CurvilinearCoordinateSystem::computeAndSetCurvature,
           "digits"_a = 8,
           "Automatically computes and sets the curvature information for the "
           "reference path."
           "\n\n:param digits:  no. of decimal points for curvature value (default 8)")

#if ENABLE_SERIALIZER
      .def("__getstate__",
          [](const geometry::CurvilinearCoordinateSystem &obj) -> std::string {
            // Return a string that fully encodes the state of the object
            std::ostringstream obj_dump;
            obj.serialize(obj_dump);
            return obj_dump.str();
          })
      .def("__setstate__",
          [](geometry::CurvilinearCoordinateSystem &obj, const std::string &str_in) {
            // Create a new C++ instance
            std::istringstream stream_in(str_in);
            auto clcs = std::const_pointer_cast<
                        geometry::CurvilinearCoordinateSystem>(geometry::CurvilinearCoordinateSystem::deserialize(stream_in));
            if (clcs == nullptr) {
              throw std::invalid_argument("pickle error - invalid input");
            }
            // We can safely move the data out of the shared_ptr, since no one else has access to it
            new (&obj) geometry::CurvilinearCoordinateSystem(std::move(*clcs));
            // reset the shared_ptr to avoid dangling references
            clcs.reset();
          })

#endif
      ;
}