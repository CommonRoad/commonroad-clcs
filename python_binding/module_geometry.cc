#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <Eigen/Dense>
#include <optional>
#include <vector>

#include "geometry/curvilinear_coordinate_system.h"
#include "geometry/segment.h"
#include "geometry/util.h"

namespace py = pybind11;

void init_module_geometry(py::module &m);

using RowMatrixXd = geometry::RowMatrixXd;

PYBIND11_MODULE(pycrccosy, m) {
  init_module_geometry(m);

  py::module mutil_geom = m.def_submodule(
      "Util",
      "Util is a submodule of pycrccosy containing auxiliary functions");

  mutil_geom.def("resample_polyline",
                 [](Eigen::Ref<const RowMatrixXd> polyline, double step) {
                   RowMatrixXd ret;
                   geometry::util::resample_polyline(polyline, step, ret);
                   return ret;
                 });

  mutil_geom.def("chaikins_corner_cutting",
                 [](Eigen::Ref<const RowMatrixXd> polyline, int refinements) {
                   RowMatrixXd ret;
                   geometry::util::chaikins_corner_cutting(polyline, refinements, ret);
                   return ret;
                 });

  mutil_geom.def("lane_riesenfeld_subdivision",
                 [](Eigen::Ref<const RowMatrixXd> polyline, int degree, int refinements) {
                    RowMatrixXd ret_polyline;
                    geometry::util::lane_riesenfeld_subdivision(polyline,
                                                                degree,
                                                                refinements,
                                                                ret_polyline);
                    return ret_polyline;
                });


  mutil_geom.def("compute_pathlength",
                 py::overload_cast<const geometry::EigenPolyline&>(&geometry::util::computePathlength),
                 py::arg("polyline"));

  mutil_geom.def("compute_curvature",
                 py::overload_cast<const geometry::EigenPolyline&>(&geometry::util::computeCurvature),
                 py::arg("polyline"));

  mutil_geom.def("compute_curvature",
                 py::overload_cast<const geometry::EigenPolyline&, int>(&geometry::util::computeCurvature),
                 py::arg("polyline"),
                 py::arg("digits"));

  mutil_geom.def("get_inflection_point_idx",
                 [](geometry::EigenPolyline polyline, int digits) {
                    return geometry::util::getInflectionPointsIdx(polyline, digits);
                });

  mutil_geom.def("intersection_segment_segment",
                 [](Eigen::Vector2d l1_pt1, Eigen::Vector2d l1_pt2,
                    Eigen::Vector2d l2_pt1, Eigen::Vector2d l2_pt2) {

                   Eigen::Vector2d intersection_point;

                   bool intersect = geometry::util::intersectionSegmentSegment(l1_pt1, l1_pt2,
                                                                               l2_pt1, l2_pt2,
                                                                               intersection_point);

                   py::tuple return_val(2);
                   return_val[0] = py::cast(intersect);
                   if (intersect) {
                    return_val[1] = py::cast(intersection_point);
                   } else {
                    return_val[1] = py::cast<py::none>(Py_None);
                   }

                   return return_val;
                 },
                 "Checks if two segments intersect; if yes, returns their intersection point"
                 "\n\n:param l1_pt1: start point of first segment"
                 "\n\n:param l1_pt2: end point of first segment"
                 "\n\n:param l2_pt1: start point of second segment"
                 "\n\n:param l2_pt2: end point of second segment"
                 "\n\n:return: Tuple with boolean for intersection and intersection point");


}

void init_module_geometry(py::module &m) {

  // Bind custom exceptions
  py::register_exception<geometry::CurvilinearProjectionDomainLongitudinalError>
          (m, "CurvilinearProjectionDomainLongitudinalError");
  py::register_exception<geometry::CurvilinearProjectionDomainLateralError>
          (m, "CurvilinearProjectionDomainLateralError");
  py::register_exception<geometry::CartesianProjectionDomainError>
          (m, "CartesianProjectionDomainError");

  py::class_<geometry::Segment>(m, "Segment")
      .def(py::init<const Eigen::Vector2d &, Eigen::Vector2d &,
                    Eigen::Vector2d &, Eigen::Vector2d &>(),
           ":param p_1: start point of the segment\n:param p_2: end point of "
           "the segment\n:param t_1: tangent vector at the start of the "
           "segment\n:param t_2: tangent vector at the end of the segment")

      .def_property_readonly("pt_1", &geometry::Segment::pt_1,
                             ":return: start point of segment in Cartesian coordinates")
      .def_property_readonly("pt_2", &geometry::Segment::pt_2,
                             ":return: end point of segment in Cartesian coordinates")
      .def_property_readonly("length", &geometry::Segment::length,
                                ":return: segment length")
      .def_property_readonly("normal_segment_start", &geometry::Segment::normalSegmentStart,
                             ":return: normal vector at the start of the segment")
      .def_property_readonly("normal_segment_end", &geometry::Segment::normalSegmentEnd,
                             ":return: normal vector at the end of the segment")
      .def_property_readonly("tangent_segment_start", &geometry::Segment::tangentSegmentStart,
                             ":return: tangent vector at the start of the segment")
      .def_property_readonly("tangent_segment_end", &geometry::Segment::tangentSegmentEnd,
                             ":return: tangent vector at the end of the segment");

  py::class_<geometry::CurvilinearCoordinateSystem,
             std::shared_ptr<geometry::CurvilinearCoordinateSystem>>(
      m, "CurvilinearCoordinateSystem")
      .def(py::init<const geometry::EigenPolyline &>())
      .def(py::init<const geometry::EigenPolyline &, const double,
                    const double>())
      .def(
          py::init<const geometry::EigenPolyline &, const double, const double,
                   const double>())
      .def(
          py::init<const geometry::EigenPolyline &, const double, const double,
                   const double, const int>(),

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
    "reference path to enable the conversion of the points near the beginning and the end of the reference path,"
    "eps2 defaults to 1e-4."
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
           [](geometry::CurvilinearCoordinateSystem &cosy){
               std::vector<geometry::Segment> vec_segment_copy;
               vec_segment_copy.reserve(cosy.getSegmentList().size());
               for (const auto &ptr : cosy.getSegmentList()) {
                   vec_segment_copy.push_back(geometry::Segment(*ptr));
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
           "Transforms a point in the curvilinear coordinate frame to the "
           "global coordinate frame.\n\n:param s: longitudinal "
           "coordinate\n\n:param l: lateral coordinate\n\n:Returns: point in "
           "global coordinates")

      .def("cartesian_point_inside_projection_domain",
           &geometry::CurvilinearCoordinateSystem::cartesianPointInProjectionDomain,
           "Validates if a point in global coordinates is within the unique "
           "projection domain.\n\n:param x: x-coordinate in the Cartesian coordinate system"
           "\n\n:param y: y-coordinate in the Cartesian coordinate system\n\n:Returns: True or False")

      .def("curvilinear_point_inside_projection_domain",
           [](geometry::CurvilinearCoordinateSystem &cosy,
              const double s, const double l){
            std::tuple<bool, bool> ret_tuple = cosy.curvilinearPointInProjectionDomain(s, l);
            bool ret_val{false};
            if(std::get<0>(ret_tuple) && std::get<1>(ret_tuple)) {
                ret_val = true;
                return py::cast(ret_val);
            } else {
                ret_val = false;
                return py::cast(ret_val);
            }
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
           [](geometry::CurvilinearCoordinateSystem &cosy,
              const std::vector<geometry::EigenPolyline> &polygons,
              const std::vector<int> groups_of_polygons,
              const int num_omp_threads) {
             std::vector<geometry::EigenPolyline> polygons_in_projection_domain;
             std::vector<int> groups_of_polygons_in_projection_domain;
             cosy.determineSubsetsOfMultiPolygonsWithinProjectionDomain(
                 polygons, groups_of_polygons, num_omp_threads,
                 polygons_in_projection_domain,
                 groups_of_polygons_in_projection_domain);
             py::tuple return_val(2);
             return_val[0] = py::cast(polygons_in_projection_domain);
             return_val[1] = py::cast(groups_of_polygons_in_projection_domain);
             return return_val;
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
           [](geometry::CurvilinearCoordinateSystem &cosy, double x, double y) {
             Eigen::Vector2d curvilinear_coord =
                 cosy.convertToCurvilinearCoords(x, y);
             return py::cast(curvilinear_coord);
           },
           "Transforms a Cartesian point to the curvilinear frame.\n\n:param "
           "x: x-coordinate in the Cartesian coordinate system\n\n:param y: "
           "y-coordinate in the Cartesian coordinate system\n\n:return: point "
           "in the curvilinear frame.")

      .def("convert_to_curvilinear_coords_and_get_segment_idx",
           [](geometry::CurvilinearCoordinateSystem &cosy, double x, double y) {
             int idx = -1;
             Eigen::Vector2d tmp =
                 cosy.convertToCurvilinearCoordsAndGetSegmentIdx(x, y, idx);
             std::vector<double> pos;
             pos.push_back(tmp(0));
             pos.push_back(tmp(1));
             py::list out;
             out.append(py::array(2, pos.data()));
             out.append(idx);
             return out;
           },
           "Transforms a Cartesian point to the curvilinear frame and returns "
           "the segment index, in which the point is contained.\n\n:param x: "
           "x-coordinate in the Cartesian coordinate system\n\n:param y: "
           "y-coordinate in the Cartesian coordinate system\n\n:return: (point "
           "in the curvilinear frame, segment index in which the point is "
           "contained)")

      .def(
          "convert_list_of_polygons_to_curvilinear_coords_and_rasterize",
          [](geometry::CurvilinearCoordinateSystem &cosy,
             const std::vector<geometry::EigenPolyline> &polygons,
             const std::vector<int> groups_of_polygons, int num_polygon_groups,
             int num_omp_threads) {
            std::vector<std::vector<geometry::EigenPolyline>>
                transformed_polygon;
            std::vector<std::vector<geometry::EigenPolyline>>
                transformed_polygon_rasterized;

            cosy.convertListOfPolygonsToCurvilinearCoordsAndRasterize(
                polygons, groups_of_polygons, num_polygon_groups,
                num_omp_threads, transformed_polygon,
                transformed_polygon_rasterized);

            py::tuple return_val(2);
            return_val[0] = py::cast(transformed_polygon);
            return_val[1] = py::cast(transformed_polygon_rasterized);
            return return_val;
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
           [](geometry::CurvilinearCoordinateSystem &cosy,
              const geometry::EigenPolyline &polygon) {
             // create output polygon
             std::vector<geometry::EigenPolyline> transformed_polygon;

             // call cpp method for converting polygon
             cosy.convertPolygonToCurvilinearCoords(polygon, transformed_polygon);

             // return
             return py::cast(transformed_polygon);
           },
           "Transforms a polygon to the curvilinear coordinate system.")

      .def("convert_rectangle_to_cartesian_coords",
           [](geometry::CurvilinearCoordinateSystem &cosy, double s_lo,
              double s_hi, double l_lo, double l_hi) {
             std::vector<geometry::EigenPolyline> triangle_mesh;
             geometry::EigenPolyline transformed_rectangle =
                 cosy.convertRectangleToCartesianCoords(s_lo, s_hi, l_lo, l_hi,
                                                        triangle_mesh);

             py::tuple return_val(2);
             return_val[0] = py::cast(transformed_rectangle);
             return_val[1] = py::cast(triangle_mesh);
             return return_val;
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

      .def("compute_and_set_curvature", [](geometry::CurvilinearCoordinateSystem &cosy, const int digits){
               return cosy.computeAndSetCurvature(digits);
           },
           py::arg("digits") = 8,
           "Automatically computes and sets the curvature information for the "
           "reference path."
           "\n\n:param digits:  no. of decimal points for curvature value (default 8)")

#if ENABLE_SERIALIZER
      .def(py::pickle(
          [](const geometry::CurvilinearCoordinateSystem
                 &obj) {  // __getstate__
            /* Return a tuple that fully encodes the state of the object */
            py::list ret;
            std::string dumped_obj;
            std::ostringstream obj_dump;
            obj.serialize(obj_dump);
            dumped_obj = obj_dump.str();
            ret.append(py::cast(dumped_obj));
            return py::make_tuple(ret);
          },
          [](py::tuple t) {  // __setstate__
                             /* Create a new C++ instance */
            std::string str_in;
            if (t.size() != 1)
              throw std::invalid_argument("pickle error - invalid input");
            py::list list_in;
            list_in = py::object(t[0]);
            if (list_in.size() != 1)
              throw std::invalid_argument("pickle error - invalid input");
            str_in = list_in[0].cast<std::string>();
            std::istringstream stream_in(str_in);
            geometry::CurvilinearCoordinateSystemConstPtr c =
                geometry::CurvilinearCoordinateSystem::deserialize(stream_in);
            if (c.get() == 0) {
              throw std::invalid_argument("pickle error - invalid input");
            } else {
              geometry::CurvilinearCoordinateSystemPtr res =
                  std::const_pointer_cast<
                      geometry::CurvilinearCoordinateSystem>(c);
              return res;
            }
          }))

#endif
      ;
}