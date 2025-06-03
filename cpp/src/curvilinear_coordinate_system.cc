#include "geometry/curvilinear_coordinate_system.h"
#include <chrono>

#if ENABLE_SERIALIZER
#include "geometry/serialize/public/serialize_public.h"
#endif

namespace geometry {

CurvilinearCoordinateSystem::CurvilinearCoordinateSystem(
    const EigenPolyline& reference_path, double default_projection_domain_limit,
    double eps, double eps2, const std::string &log_level, int method) :
    default_projection_domain_limit_(default_projection_domain_limit),
    eps_(eps),
    eps2_(eps2),
    method_(method) {

  // initialize logger
  CLCSLogger::init(log_level);

  // pre-check reference path
  if (reference_path.size() < 3) {
    throw InvalidReferencePathError();
  }

  // initialize path segments from input reference path
  this->path_segments_ptr = std::make_unique<PathSegments>(reference_path, eps2);

  // store original ref path (due to required extensions)
  this->reference_path_original_ = reference_path;

  // compute and initialize curvature
  this->computeAndSetCurvature();

  // initialize projection domain handler
  this->projection_domain_ptr = std::make_unique<ProjectionDomain>(this->path_segments_ptr->getSegmentList(),
                                                                   this->curvature_vec_,
                                                                   this->curvature_radius_vec_,
                                                                   this->path_segments_ptr->getSegmentLongitudinalCoord(),
                                                                   this->path_segments_ptr->getLength(),
                                                                   default_projection_domain_limit,
                                                                   eps, method);

  // pre-compute projection axis for segments
  this->path_segments_ptr->computeBestProjectionAxisForSegments(this->upperProjectionDomainBorder(),
                                                                this->lowerProjectionDomainBorder());

  // compute partitions of reference path at inflection points
  util::computePathPartitions(this->referencePath(), this->reference_path_partitions_);

  // log info
  CLCSLogger::getLogger()->info("Initialization of CLCS done");
}

EigenPolyline CurvilinearCoordinateSystem::referencePath() const {
  return this->path_segments_ptr->getReferencePath();
}

EigenPolyline CurvilinearCoordinateSystem::referencePathOriginal() const {
  return this->reference_path_original_;
}

std::vector<EigenPolyline> CurvilinearCoordinateSystem::referencePathPartitions() const {
    return this->reference_path_partitions_;
}

EigenPolyline CurvilinearCoordinateSystem::projectionDomainBorder() const {
  return this->projection_domain_ptr->projectionDomainBorder();
}

EigenPolyline CurvilinearCoordinateSystem::curvilinearProjectionDomainBorder() const {
  return this->projection_domain_ptr->curvilinearProjectionDomainBorder();
}

EigenPolyline CurvilinearCoordinateSystem::upperProjectionDomainBorder() const {
    return this->projection_domain_ptr->upperProjectionDomainBorder();
}

EigenPolyline CurvilinearCoordinateSystem::lowerProjectionDomainBorder() const {
    return this->projection_domain_ptr->lowerProjectionDomainBorder();
}

double CurvilinearCoordinateSystem::length() const {
    return this->path_segments_ptr->getLength();
}

const std::vector<double> &CurvilinearCoordinateSystem::segmentsLongitudinalCoordinates() const {
  return this->path_segments_ptr->getSegmentLongitudinalCoord();
}

const std::vector<std::unique_ptr<Segment>> &CurvilinearCoordinateSystem::getSegmentList() const {
    return this->path_segments_ptr->getSegmentList();
}

void CurvilinearCoordinateSystem::setLoggingLevel(const std::string &log_level) {
    CLCSLogger::init(log_level);
}

void CurvilinearCoordinateSystem::setCurvature(std::vector<double> curvature) {
  std::size_t num_path_points = this->path_segments_ptr->getNumPathPoints();

  if (curvature.size() != num_path_points) {
    throw std::invalid_argument(
        "<CurvilinearCoordinateSystem/setCurvature> Curvature values must be "
        "given for each segment. "
        "Length of curvature vector: " + std::to_string(curvature.size()) +
        "Number of segments: " + std::to_string(num_path_points));
  }

  // set curvature vector
  this->curvature_vec_ = curvature;

  for (int i = 0; i < curvature.size(); i++) {
      // set curvature value at longitudinal position of reference path
      this->curvature_[this->segmentsLongitudinalCoordinates()[i]] = curvature[i];

      // set curvature radius vector
      if (fabs(this->curvature_vec_[i]) <= 1e-8) {
          this->curvature_radius_vec_.push_back(1e8);
      } else {
          this->curvature_radius_vec_.push_back(1 / fabs(this->curvature_vec_[i]));
      }
  }

  // set max and min curvature
  this->max_curvature_ = *std::max_element(curvature.begin(), curvature.end());
  this->min_curvature_ = *std::min_element(curvature.begin(), curvature.end());

  // set max and min curvature radius
  this->max_curvature_radius_ = *std::max_element(this->curvature_radius_vec_.begin(),
                                                  this->curvature_radius_vec_.end());
  this->min_curvature_radius_ = *std::min_element(this->curvature_radius_vec_.begin(),
                                                  this->curvature_radius_vec_.end());
}

std::tuple<double, double> CurvilinearCoordinateSystem::curvatureRange(
    double s_min, double s_max) const {
  // get the first longitudinal position which is lower or equal than s_min
  auto it_min = this->curvature_.upper_bound(s_min);
  if (it_min == this->curvature_.begin()) {
    std::cout << "s_min: " << s_min << std::endl;
    throw CurvilinearProjectionDomainLongitudinalError();
  } else {
    it_min--;
  }

  // get the first longitudinal position which higher or equal than s_max
  auto it_max = this->curvature_.lower_bound(s_max);
  if (it_max == this->curvature_.end()) {
    std::cout << "s_max: " << s_max << std::endl;
    // first and all other longitudinal positions are higher than s_max
    throw CurvilinearProjectionDomainLongitudinalError();
  } else {
    it_max++;
  }

  std::vector<double> curvature_range;
  for (auto it = it_min; it != it_max; ++it) {
    curvature_range.push_back((*it).second);
  }
  std::sort(curvature_range.begin(), curvature_range.end());
  return std::make_tuple(curvature_range.front(), curvature_range.back());
}

std::vector<double> CurvilinearCoordinateSystem::curvatureVector(void) const {
    return curvature_vec_;
}

std::vector<double> CurvilinearCoordinateSystem::curvatureRadiusVector(void) const {
    return curvature_radius_vec_;
}

double CurvilinearCoordinateSystem::maximumCurvatureRadius() const {
  if (std::isnan(this->max_curvature_radius_)) {
    throw std::invalid_argument(
        "<CurvilinearCoordinateSystem/getMaximumCurvatureRadius> Maximum "
        "curvature radius must be set first.");
  }
  return this->max_curvature_radius_;
}

double CurvilinearCoordinateSystem::minimumCurvatureRadius() const {
  if (std::isnan(this->min_curvature_radius_)) {
    throw std::invalid_argument(
        "<CurvilinearCoordinateSystem/getMinimumCurvatureRadius> Minimum "
        "curvature radius must be set first.");
  }
  return this->min_curvature_radius_;
}

double CurvilinearCoordinateSystem::maximumCurvature() const {
  if (std::isnan(this->max_curvature_)) {
    throw std::invalid_argument(
        "<CurvilinearCoordinateSystem/getMaximumCurvature> Maximum curvature "
        "must be set first.");
  }
  return this->max_curvature_;
}

double CurvilinearCoordinateSystem::minimumCurvature() const {
  if (std::isnan(this->min_curvature_)) {
    throw std::invalid_argument(
        "<CurvilinearCoordinateSystem/getMinimumCurvature> Minimum curvature "
        "must be set first.");
  }
  return this->min_curvature_;
}

Eigen::Vector2d CurvilinearCoordinateSystem::normal(double s) const {
  return this->path_segments_ptr->normal(s);
}

Eigen::Vector2d CurvilinearCoordinateSystem::tangent(double s) const {
  return this->path_segments_ptr->tangent(s);
}

Eigen::Vector2d CurvilinearCoordinateSystem::convertToCartesianCoords(
    double s, double l, bool check_proj_domain) const {
    // get idx of segment corresponding to longitudinal coordinate s
    int idx = this->path_segments_ptr->findSegmentIndex(s);

    // check if point is within curvilinear projection domain
    if (check_proj_domain) {
        bool in_longitudinal_bounds;
        bool in_lateral_bounds;
        std::tie(in_longitudinal_bounds, in_lateral_bounds) =
                this->curvilinearPointInProjectionDomain(idx, s, l);

        if (!in_longitudinal_bounds) {
            throw CurvilinearProjectionDomainLongitudinalError();
        }

        if (!in_lateral_bounds) {
            throw CurvilinearProjectionDomainLateralError();
        }
    } else {
        CLCSLogger::getLogger()->debug("<convertToCartesianCoords()> Converting without projection domain check");
    }

    const auto &segment = this->path_segments_ptr->getSegmentByID(idx);

    return segment->convertToCartesianCoords(
            s - this->segmentsLongitudinalCoordinates()[idx], l);
}

Eigen::Vector2d CurvilinearCoordinateSystem::convertToCurvilinearCoords(
    double x, double y, bool check_proj_domain) const {
  int segment_idx = -1;
  return this->convertToCurvilinearCoordsAndGetSegmentIdx(x, y, segment_idx, check_proj_domain);
}

Eigen::Vector2d
CurvilinearCoordinateSystem::convertToCurvilinearCoordsAndGetSegmentIdx(
    double x, double y, int &segment_idx, bool check_proj_domain) const {

  // check if in projection domain
  if (check_proj_domain) {
      bool is_in_projection_domain = this->cartesianPointInProjectionDomain(x, y);

      if (!is_in_projection_domain) {
          throw CartesianProjectionDomainError();
      }
  } else {
      CLCSLogger::getLogger()->debug("<convertToCurvilinearCoords()> Converting without projection domain check");
  }

  /* Determine segment candidates:
   * Loop over all segments and convert to curvilinear Coordinates for each segment
   * Check if interpolation parameter lambda is in [0, 1], if yes: add segment to candidate list
   * Sort candidates according to lateral curvilinear coordinate (d) of converted point
   * We choose the segment and converted point with smallest d value -> closest point projection
   */
  std::vector<std::pair<Eigen::Vector2d, int>> candidates;
  double lambda;
  std::size_t num_segments = this->path_segments_ptr->getNumSegments();

  for (int i = 0; i < num_segments; i++) {
    const auto &segment = this->path_segments_ptr->getSegmentByID(i);
    Eigen::Vector2d curvilinear_point =
        segment->convertToCurvilinearCoords(x, y, lambda);
    if (std::isgreaterequal(lambda + 10e-8, 0.0) &&
        std::islessequal(lambda - 10e-8, 1.0)) {
      candidates.push_back(
          std::pair<Eigen::Vector2d, int>(curvilinear_point, i));
    }
  }

  if (candidates.empty()) {
    throw CartesianProjectionDomainError();
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const std::pair<Eigen::Vector2d, int> &a,
               const std::pair<Eigen::Vector2d, int> &b) {
              return std::abs(a.first(1)) < std::abs(b.first(1));
            });

  segment_idx = candidates[0].second;
  return candidates[0].first +
         Eigen::Vector2d(this->segmentsLongitudinalCoordinates()[segment_idx], 0.);
}


EigenPolyline
CurvilinearCoordinateSystem::convertListOfPointsToCurvilinearCoords(
        const EigenPolyline &points, int num_omp_threads) const {
    std::vector<EigenPolyline> points_in(1);
    for (const auto p : points) {
        if (this->cartesianPointInProjectionDomain(p.x(), p.y())) {
            points_in[0].push_back(p);
        }
    }

    std::vector<std::vector<std::tuple<int, double, double>>>
            transformed_coordinates_and_segment_idx;
    transformed_coordinates_and_segment_idx =
            this->convertToCurvilinearCoords(points_in, num_omp_threads);

    EigenPolyline transformed_points;
    if (transformed_coordinates_and_segment_idx.size() == 1) {
        for (const auto p : transformed_coordinates_and_segment_idx[0]) {
            transformed_points.push_back(
                    Eigen::Vector2d(std::get<1>(p), std::get<2>(p)));
        }
    }
    return transformed_points;
}

EigenPolyline CurvilinearCoordinateSystem::convertListOfPointsToCartesianCoords(const EigenPolyline &points,
int num_omp_threads) const {

    // settings for OMP
    omp_set_dynamic(0);
    omp_set_num_threads(num_omp_threads);
    omp_lock_t writelock;
    omp_init_lock(&writelock);

    // for each point in the polyline
    EigenPolyline group_of_cartesian_points;
    group_of_cartesian_points.resize(points.size());
#pragma omp parallel
    {
#pragma omp for nowait
    for (int point_index = 0; point_index < points.size();
         point_index++) {
	    auto point=points[point_index];
            // get for each point the coordinates
            double s_coordinate = point.x();
            double l_coordinate = point.y();
            int segment_index = this->path_segments_ptr->findSegmentIndex(s_coordinate);

            // check if point is within curvilinear projection domain
            bool in_longitudinal_bounds;
            bool in_lateral_bounds;
            std::tie(in_longitudinal_bounds, in_lateral_bounds) = this->curvilinearPointInProjectionDomain(
                    segment_index, s_coordinate, l_coordinate);

            if (!in_longitudinal_bounds) {
                throw CurvilinearProjectionDomainLongitudinalError();
            }

            if (!in_lateral_bounds) {
                throw CurvilinearProjectionDomainLateralError();
            }
            const auto &segment = this->path_segments_ptr->getSegmentByID(segment_index);
            Eigen::Vector2d cartesian_coord = segment->convertToCartesianCoords(
                    s_coordinate - this->segmentsLongitudinalCoordinates()[segment_index], l_coordinate);

            omp_set_lock(&writelock);
            group_of_cartesian_points[point_index] = cartesian_coord;
            omp_unset_lock(&writelock);
        }
    }
    omp_destroy_lock(&writelock);

    return group_of_cartesian_points;
}

EigenPolyline CurvilinearCoordinateSystem::convertRectangleToCartesianCoords(
    double s_lo, double s_hi, double l_lo, double l_hi,
    std::vector<EigenPolyline> &triangle_mesh) const {
  int idx_lo = this->path_segments_ptr->findSegmentIndex(s_lo);
  int idx_hi = this->path_segments_ptr->findSegmentIndex(s_hi);
  triangle_mesh.reserve((idx_hi - idx_lo) * 2);

  for (int idx = idx_lo; idx <= idx_hi; idx++) {
    const auto &segment = this->path_segments_ptr->getSegmentByID(idx);
    double segment_s_lo =
        std::max(0.0, s_lo - this->segmentsLongitudinalCoordinates()[idx]);
    double segment_s_hi = std::min(
        segment->length_, s_hi - this->segmentsLongitudinalCoordinates()[idx]);
    auto tmp = segment->convertRectangleToCartesianCoords(
        segment_s_lo, segment_s_hi, l_lo, l_hi);
    triangle_mesh.insert(triangle_mesh.end(), tmp.begin(), tmp.end());
  }
  // Construct vertices of polygon from triangle mesh
  std::vector<Eigen::Vector2d> lower_boundary;
  lower_boundary.reserve(triangle_mesh.size() / 2 + 1);
  std::vector<Eigen::Vector2d> upper_boundary;
  upper_boundary.reserve(triangle_mesh.size() / 2 + 1);

  for (auto it = triangle_mesh.begin(); it != triangle_mesh.end(); it += 2) {
    lower_boundary.push_back((*it)[0]);
    upper_boundary.push_back((*it)[2]);
  }

  EigenPolyline triangle = triangle_mesh.back();
  lower_boundary.push_back(triangle[0]);
  upper_boundary.push_back(triangle[2]);

  EigenPolyline polygon;
  polygon.insert(polygon.end(), upper_boundary.begin(), upper_boundary.end());
  polygon.insert(polygon.end(), lower_boundary.rbegin(), lower_boundary.rend());
  if (!upper_boundary.front().isApprox(lower_boundary.front())) {
    polygon.push_back(upper_boundary.front());
  }

  return polygon;
}

void CurvilinearCoordinateSystem::convertPolygonToCurvilinearCoords(
    const EigenPolyline &polygon,
    std::vector<EigenPolyline> &transformed_polygon) const {
  std::vector<std::vector<EigenPolyline>> transformed_polygons;
  std::vector<std::vector<EigenPolyline>> transformed_polygons_rasterized;

  std::vector<EigenPolyline> polygon_in(1, polygon);
  std::vector<int> groups(1, 0);
  this->convertListOfPolygonsToCurvilinearCoordsAndRasterize(
      polygon_in, groups, 1, 4, transformed_polygons,
      transformed_polygons_rasterized);
  if (transformed_polygons.size() != 0) {
    transformed_polygon = transformed_polygons[0];
  }
}

void CurvilinearCoordinateSystem::
    convertListOfPolygonsToCurvilinearCoordsAndRasterize(
        const std::vector<EigenPolyline> &polygons,
        const std::vector<int> groups_of_polygons, int num_polygon_groups,
        int num_omp_threads,
        std::vector<std::vector<EigenPolyline>> &transformed_polygons,
        std::vector<std::vector<EigenPolyline>>
            &transformed_polygons_rasterized) const {
  omp_set_dynamic(0);
  omp_set_num_threads(num_omp_threads);
  omp_lock_t writelock;
  omp_init_lock(&writelock);

  transformed_polygons.resize(num_polygon_groups);
  transformed_polygons_rasterized.resize(num_polygon_groups);

  // intersect polygons with projection domain
  std::vector<EigenPolyline> clipped_polygon_all;
  std::vector<int> clipped_polygon_groups_all;
  this->determineSubsetsOfMultiPolygonsWithinProjectionDomain(
      polygons, groups_of_polygons, num_omp_threads, clipped_polygon_all,
      clipped_polygon_groups_all);

  // transform all polygons to curvilinear coordinates
  std::vector<std::vector<std::tuple<int, double, double>>>
      poly_curvil_coordinates;
  poly_curvil_coordinates =
      this->convertToCurvilinearCoords(clipped_polygon_all, num_omp_threads);

#pragma omp parallel
  {
    std::vector<std::vector<EigenPolyline>> transformed_polygon_thread_all(
        num_polygon_groups);
    std::vector<std::vector<EigenPolyline>>
        transformed_polygon_rasterized_thread_all(num_polygon_groups);

#pragma omp for nowait
    for (std::vector<EigenPolyline>::const_iterator polygon_it =
             clipped_polygon_all.begin();
         polygon_it < clipped_polygon_all.end(); ++polygon_it) {
      std::vector<EigenPolyline> transformed_polygon_thread;
      std::vector<EigenPolyline> transformed_polygon_rasterized_thread;
      std::set<int> segment_indices_thread;

      int cur_poly_index = polygon_it - clipped_polygon_all.begin();

      // add further points for each transition to a new segment
      EigenPolyline transformed_polygon;
      std::set<int> indices;
      poly_curvil_coordinates[cur_poly_index].push_back(
          poly_curvil_coordinates[cur_poly_index].front());
      bool success = this->addPointsAtSegmentTransition(
          *polygon_it, poly_curvil_coordinates[cur_poly_index],
          transformed_polygon, indices);

      if (success) {
        success = transformed_polygon.size() > 0;
        transformed_polygon_thread.push_back(transformed_polygon);
        segment_indices_thread.insert(indices.begin(), indices.end());
      }

      // approximate transformed polygons with axis-aligned rectangles
      int cur_poly_group = clipped_polygon_groups_all[cur_poly_index];
      if (success && transformed_polygon_thread.size() > 0) {
        this->rasterizeListOfTransformedPolygonsInProjectionDomain(
            transformed_polygon_thread, segment_indices_thread,
            transformed_polygon_rasterized_thread);

        transformed_polygon_thread_all[cur_poly_group].insert(
            transformed_polygon_thread_all[cur_poly_group].end(),
            std::make_move_iterator(transformed_polygon_thread.begin()),
            std::make_move_iterator(transformed_polygon_thread.end()));

        transformed_polygon_rasterized_thread_all[cur_poly_group].insert(
            transformed_polygon_rasterized_thread_all[cur_poly_group].end(),
            std::make_move_iterator(
                transformed_polygon_rasterized_thread.begin()),
            std::make_move_iterator(
                transformed_polygon_rasterized_thread.end()));
      }
    }

    omp_set_lock(&writelock);
    for (int i = 0; i < num_polygon_groups; i++) {
      transformed_polygons[i].insert(
          transformed_polygons[i].end(),
          std::make_move_iterator(transformed_polygon_thread_all[i].begin()),
          std::make_move_iterator(transformed_polygon_thread_all[i].end()));

      transformed_polygons_rasterized[i].insert(
          transformed_polygons_rasterized[i].end(),
          std::make_move_iterator(
              transformed_polygon_rasterized_thread_all[i].begin()),
          std::make_move_iterator(
              transformed_polygon_rasterized_thread_all[i].end()));
    }
    omp_unset_lock(&writelock);
  }

  omp_destroy_lock(&writelock);
}

bool CurvilinearCoordinateSystem::addPointsAtSegmentTransition(
    const EigenPolyline &polygon,
    const std::vector<std::tuple<int, double, double>>
        &curvilinear_coordinates_and_segment_idx,
    EigenPolyline &curvilinear_polygon, std::set<int> &segment_indices) const {
  if (polygon.empty()) {
    return false;
  }

  EigenPolyline polygon_vertices(polygon.begin(), polygon.end());
  int previous_segment_idx;
  int segment_idx;
  Eigen::Vector2d curvilinear_point;

  previous_segment_idx =
      std::get<0>(curvilinear_coordinates_and_segment_idx[0]);
  curvilinear_point =
      Eigen::Vector2d(std::get<1>(curvilinear_coordinates_and_segment_idx[0]),
                      std::get<2>(curvilinear_coordinates_and_segment_idx[0]));
  curvilinear_polygon.push_back(curvilinear_point);
  segment_indices.insert(previous_segment_idx);
  polygon_vertices.push_back(polygon_vertices.front());

  for (int i = 1; i < polygon_vertices.size(); i++) {
    segment_idx = std::get<0>(curvilinear_coordinates_and_segment_idx[i]);
    curvilinear_point = Eigen::Vector2d(
        std::get<1>(curvilinear_coordinates_and_segment_idx[i]),
        std::get<2>(curvilinear_coordinates_and_segment_idx[i]));
    segment_indices.insert(segment_idx);
    // check if new point lies in a different segment than the previous point,
    // if yes, compute intermediate points for each segment in between
    if (segment_idx == previous_segment_idx) {
      curvilinear_polygon.push_back(curvilinear_point);
    } else {
      std::list<int> indices_range =
          this->path_segments_ptr->determineIndicesRange(segment_idx, previous_segment_idx);
      segment_indices.insert(indices_range.begin(), indices_range.end());
      this->path_segments_ptr->interpolatePointsBetweenSegments(polygon_vertices[i - 1],
                                                                polygon_vertices[i],
                                                                indices_range,
                                                                this->default_projection_domain_limit_,
                                                                curvilinear_polygon);
      if (!curvilinear_polygon.back().isApprox(curvilinear_point, 10e-8)) {
        curvilinear_polygon.push_back(curvilinear_point);
      }
    }
    previous_segment_idx = segment_idx;
  }
  if (curvilinear_polygon.front().isApprox(curvilinear_polygon.back(), 10e-8)) {
    curvilinear_polygon.pop_back();
  }
  return true;
}

void CurvilinearCoordinateSystem::
    rasterizeListOfTransformedPolygonsInProjectionDomain(
        const std::vector<EigenPolyline> &transformed_polygons,
        const std::set<int> &segment_indices,
        std::vector<EigenPolyline> &transformed_polygons_rasterized) const {
  for (const auto &polygon : transformed_polygons) {
    std::vector<EigenPolyline> rectangle_list;
    this->rasterizeTransformedPolygonInProjectionDomain(
        polygon, segment_indices, rectangle_list);
    transformed_polygons_rasterized.insert(
        transformed_polygons_rasterized.end(), rectangle_list.begin(),
        rectangle_list.end());
  }
}

void CurvilinearCoordinateSystem::rasterizeTransformedPolygonInProjectionDomain(
    const EigenPolyline &transformed_polygon,
    const std::set<int> &segment_indices,
    std::vector<EigenPolyline> &transformed_polygon_rasterized) const {
  auto createRectangle = [](EigenPolyline points) {
    double x_min = points[0][0];
    double x_max = points[0][0];
    double y_min = points[0][1];
    double y_max = points[0][1];
    for (const auto &p : points) {
      x_min = p[0] < x_min ? p[0] : x_min;
      x_max = p[0] > x_max ? p[0] : x_max;
      y_min = p[1] < y_min ? p[1] : y_min;
      y_max = p[1] > y_max ? p[1] : y_max;
    }
    EigenPolyline rectangle{
        Eigen::Vector2d(x_min, y_min), Eigen::Vector2d(x_max, y_min),
        Eigen::Vector2d(x_max, y_max), Eigen::Vector2d(x_min, y_max)};
    return rectangle;
  };

  std::set<int> segment_indices_extended(segment_indices.begin(),
                                         segment_indices.end());
  segment_indices_extended.insert(*segment_indices.rbegin() + 1);
  if (*segment_indices.begin() > 0) {
    segment_indices_extended.insert(*segment_indices.begin() - 1);
  }

  double last_maximum_s_coord = std::numeric_limits<double>::infinity();
  std::set<int>::iterator it;
  for (it = std::next(segment_indices_extended.begin());
       it != segment_indices_extended.end(); ++it) {
    double s_min = std::min(this->segmentsLongitudinalCoordinates()[*std::prev(it)],
                            last_maximum_s_coord) -
                   10e-3;
    double s_max = this->segmentsLongitudinalCoordinates()[*it] + 10e-3;

    EigenPolyline points_in_segment;
    if ((it != segment_indices_extended.end()) &&
        (std::next(it) == segment_indices_extended.end())) {
      copy_if(transformed_polygon.begin(), transformed_polygon.end(),
              back_inserter(points_in_segment), [s_min](Eigen::Vector2d p) {
                return std::isgreaterequal(p[0], s_min);
              });
    } else {
      copy_if(transformed_polygon.begin(), transformed_polygon.end(),
              back_inserter(points_in_segment),
              [s_min, s_max](Eigen::Vector2d p) {
                return (std::isgreaterequal(p[0], s_min) &&
                        std::islessequal(p[0], s_max));
              });
    }
    if (!points_in_segment.empty()) {
      EigenPolyline rectangle = createRectangle(points_in_segment);
      transformed_polygon_rasterized.push_back(rectangle);
      last_maximum_s_coord = rectangle[1][0];
    }
  }
}

bool CurvilinearCoordinateSystem::cartesianPointInProjectionDomain(double x, double y) const {
  return this->projection_domain_ptr->cartesianPointInProjectionDomain(x, y);
}

std::tuple<bool, bool> CurvilinearCoordinateSystem::curvilinearPointInProjectionDomain(double s, double l) const {
    int segment_index = this->path_segments_ptr->findSegmentIndex(s);
    return this->curvilinearPointInProjectionDomain(segment_index, s, l);
}

std::tuple<bool, bool> CurvilinearCoordinateSystem::curvilinearPointInProjectionDomain(
        int seg_idx, double s, double l) const {
    const auto &segment = this->path_segments_ptr->getSegmentByID(seg_idx);
    const auto seg_lon_coord = this->segmentsLongitudinalCoordinates()[seg_idx];
    return this->projection_domain_ptr->curvilinearPointInProjectionDomain(segment, seg_idx, seg_lon_coord, s, l);
}

std::vector<EigenPolyline> CurvilinearCoordinateSystem::determineSubsetOfPolygonWithinProjectionDomain(
    const EigenPolyline &polygon) const {
  return this->projection_domain_ptr->determineSubsetOfPolygonWithinProjectionDomain(polygon);
}

std::vector<EigenPolyline>
CurvilinearCoordinateSystem::determineSubsetOfPolygonWithinCurvilinearProjectionDomain(
        const EigenPolyline &polygon) const {
  return this->projection_domain_ptr->determineSubsetOfPolygonWithinCurvilinearProjectionDomain(polygon);
}

// TODO move to projection_domain.h
void CurvilinearCoordinateSystem::
    determineSubsetsOfMultiPolygonsWithinProjectionDomain(
        const std::vector<EigenPolyline> &polygons,
        const std::vector<int> groups_of_polygons, const int num_omp_threads,
        std::vector<EigenPolyline> &polygons_in_projection_domain,
        std::vector<int> &groups_of_polygons_in_projection_domain) const {
  omp_set_dynamic(0);
  omp_set_num_threads(num_omp_threads);

  omp_lock_t writelock;
  omp_init_lock(&writelock);

#pragma omp parallel
  {
    std::vector<EigenPolyline> polygons_in_projection_domain_thread;
    std::vector<int> groups_of_polygons_in_projection_domain_thread;
#pragma omp for nowait
    for (std::vector<EigenPolyline>::const_iterator polygon_it =
             polygons.begin();
         polygon_it < polygons.end(); ++polygon_it) {
      std::vector<EigenPolyline> polygon_in_proj_domain =
          this->determineSubsetOfPolygonWithinProjectionDomain(*polygon_it);
      std::vector<int> polygon_groups;
      for (int i = 0; i < polygon_in_proj_domain.size(); i++) {
        polygon_groups.push_back(
            groups_of_polygons[polygon_it - polygons.begin()]);
      }

      polygons_in_projection_domain_thread.insert(
          polygons_in_projection_domain_thread.end(),
          std::make_move_iterator(polygon_in_proj_domain.begin()),
          std::make_move_iterator(polygon_in_proj_domain.end()));

      groups_of_polygons_in_projection_domain_thread.insert(
          groups_of_polygons_in_projection_domain_thread.end(),
          std::make_move_iterator(polygon_groups.begin()),
          std::make_move_iterator(polygon_groups.end()));
    }
    omp_set_lock(&writelock);
    polygons_in_projection_domain.insert(
        polygons_in_projection_domain.end(),
        std::make_move_iterator(polygons_in_projection_domain_thread.begin()),
        std::make_move_iterator(polygons_in_projection_domain_thread.end()));
    groups_of_polygons_in_projection_domain.insert(
        groups_of_polygons_in_projection_domain.end(),
        std::make_move_iterator(
            groups_of_polygons_in_projection_domain_thread.begin()),
        std::make_move_iterator(
            groups_of_polygons_in_projection_domain_thread.end()));
    omp_unset_lock(&writelock);
  }
  omp_destroy_lock(&writelock);
}

void CurvilinearCoordinateSystem::determineCurvilinearCoordinatesAndSegmentIdx(
    const std::vector<std::vector<std::tuple<int, int>>>
        &candidate_segments_of_points,
    const std::vector<Eigen::RowVectorXd,
                      Eigen::aligned_allocator<Eigen::RowVectorXd>> &s_coord,
    const std::vector<Eigen::RowVectorXd,
                      Eigen::aligned_allocator<Eigen::RowVectorXd>> &l_coord,
    int num_omp_threads,
    std::vector<std::vector<std::tuple<int, double, double>>>
        &groups_of_curvil_points) const {
  // settings for OMP
  omp_set_dynamic(0);
  omp_set_num_threads(num_omp_threads);

#pragma omp parallel
  {
#pragma omp for nowait
    for (int i = 0; i < groups_of_curvil_points.size(); i++) {
      for (int j = 0; j < groups_of_curvil_points[i].size(); j++) {
        int orig_idx = 0;
        for (int k = 0; k < i; k++) {
          orig_idx += groups_of_curvil_points[k].size();
        }
        orig_idx += j;

        int best_segment = -1;
        int best_idx = -1;
        if (candidate_segments_of_points[orig_idx].size() > 1) {
          // Case 1: there exist more than one valid segment for the point -> take the one
          // with minimal lateral coordinate
          double signed_distance = std::numeric_limits<double>::infinity();
          for (const auto el : candidate_segments_of_points[orig_idx]) {
            int segment_idx = std::get<0>(el);
            int idx = std::get<1>(el);
            if (std::fabs(l_coord[segment_idx][idx]) < signed_distance) {
              signed_distance = std::fabs(l_coord[segment_idx][idx]);
              best_segment = segment_idx;
              best_idx = idx;
            }
          }
        } else if (candidate_segments_of_points[orig_idx].size() == 1) {
          // Case 2: point only has one candidate segment
          best_segment = std::get<0>(candidate_segments_of_points[orig_idx][0]);
          best_idx = std::get<1>(candidate_segments_of_points[orig_idx][0]);
        } else {
          // Case 3: point has no candidate segment -> outside of projection domain?
          throw CartesianProjectionDomainError();
        }
        groups_of_curvil_points[i][j] =
            std::make_tuple(best_segment, s_coord[best_segment][best_idx],
                            l_coord[best_segment][best_idx]);
      }
    }
  }
}

int CurvilinearCoordinateSystem::computeAndSetCurvature(int digits) {
  // compute rounded curvature with precision
  auto curvature = geometry::util::computeCurvature(referencePath(), digits);
  std::vector<double> vec_curv;

  for (int i = 0; i < curvature.size(); i++) {
      vec_curv.push_back(curvature[i]);
  }
  this->setCurvature(vec_curv);
  return 0;
}

std::vector<std::vector<std::tuple<int, double, double>>>
CurvilinearCoordinateSystem::convertToCurvilinearCoords(
    const std::vector<EigenPolyline> &groups_of_points,
    int num_omp_threads) const {
  // settings for OMP
  omp_set_dynamic(0);
  omp_set_num_threads(num_omp_threads);
  omp_lock_t writelock;
  omp_init_lock(&writelock);

  // determine number of incoming points
  int number_of_points_in_all_groups = 0;
  for (const auto &points : groups_of_points) {
    number_of_points_in_all_groups += points.size();
  }

  // restructure incoming points
//  [x1 x2 x3 ... xN
//   y1 y2 y3 ... yN]
// We need these to find the candidate points (the closest segment to the point/s)
  Eigen::Matrix2Xd points_in(2, number_of_points_in_all_groups);
  std::vector<std::pair<double, int>> pairs_in_x;
  std::vector<std::pair<double, int>> pairs_in_y;
  std::vector<std::pair<double, int>> pairs_in_x_rotated;
  std::vector<std::pair<double, int>> pairs_in_y_rotated;

  int offset_cols = 0;
  for (const auto &points : groups_of_points) {
    for (const auto &p : points) {
      pairs_in_x.push_back(std::make_pair(p.x(), offset_cols));
      pairs_in_y.push_back(std::make_pair(p.y(), offset_cols));
      points_in.middleCols(offset_cols++, 1) = p;
    }
  }

  // rotate points 45 degree
  Eigen::Matrix2d rot_axis_matr;
//   [ cosA -sinA
//     sinA cosA ]
  rot_axis_matr << sqrt(2) / 2, sqrt(2) / 2, -sqrt(2) / 2, sqrt(2) / 2;
  Eigen::MatrixXd points_in_rotated;
  // Rot(A) * Points_In
  points_in_rotated = rot_axis_matr * points_in.eval();
  for (int i = 0; i < number_of_points_in_all_groups; i++) {
    pairs_in_x_rotated.push_back(std::make_pair(points_in_rotated(0, i), i));
    pairs_in_y_rotated.push_back(std::make_pair(points_in_rotated(1, i), i));
  }

  // sort coordinates of points in increasing order
  std::vector<std::vector<std::pair<double, int>> *> to_sort;
  to_sort.push_back(&pairs_in_x);
  to_sort.push_back(&pairs_in_x_rotated);
  to_sort.push_back(&pairs_in_y);
  to_sort.push_back(&pairs_in_y_rotated);

  // get number of segments
  std::size_t num_segments = this->path_segments_ptr->getNumSegments();

  // parallelize sorting over x, x_rotated, y, y_rotated
#pragma omp parallel
  {
#pragma omp for nowait
    for (int sort_index = 0; sort_index < 4; sort_index++) {
      std::sort(to_sort[sort_index]->begin(), to_sort[sort_index]->end());
    }
  }

  std::vector<std::vector<std::tuple<int, int>>> candidate_segments_of_points(
      number_of_points_in_all_groups, std::vector<std::tuple<int, int>>(0));
  std::vector<Eigen::RowVectorXd, Eigen::aligned_allocator<Eigen::RowVectorXd>>
      s_results(num_segments);
  std::vector<Eigen::RowVectorXd, Eigen::aligned_allocator<Eigen::RowVectorXd>>
      l_results(num_segments);

#pragma omp parallel
  {
#pragma omp for nowait
    for (int segment_index = 0; segment_index < num_segments;
         segment_index++) {
      // find points which are potentially within the current segment
      std::vector<int> candidates_indices =
              this->path_segments_ptr->computeCandidateIndices(segment_index,
                                                               pairs_in_x,
                                                               pairs_in_x_rotated,
                                                               pairs_in_y,
                                                               pairs_in_y_rotated);

      // get the coordinates of points which might be in the current segment
      Eigen::Matrix2Xd candidate_points_in_segment(2,
                                                   candidates_indices.size());
      int offset_cols = 0;
      for (auto el : candidates_indices) {
        candidate_points_in_segment.middleCols(offset_cols++, 1) =
            points_in.middleCols(el, 1);
      }

      const auto &segment = this->path_segments_ptr->getSegmentByID(segment_index);
      // rotate points to local frame of segment
      Eigen::Matrix2Xd candidate_points_in_segment_local;
      segment->rotatePointsToLocalFrame(candidate_points_in_segment,
                                        candidate_points_in_segment_local);
      // compute scaled lambdas
      Eigen::RowVectorXd scaled_lambdas;
      Eigen::RowVectorXd dividers;
      segment->computeScaledLambdas(candidate_points_in_segment_local, dividers,
                                    scaled_lambdas);
      // check validity of lambdas
      // valid (scaled) lambda is in interval [0, divider]
      Eigen::RowVectorXd valid_lambdas(candidates_indices.size());
      Eigen::RowVectorXd valid_candidates_y(candidates_indices.size());
      Eigen::Matrix2Xd valid_candidate_points_in_segment(
          2, candidates_indices.size());
      int k = 0;
      int n_cand = 0;
      for (auto orig_ind : candidates_indices) {
        if (dividers(k) > 0.) {
          if ((scaled_lambdas(k) + 10e-8 >= 0.0) &&
              (scaled_lambdas(k) - 10e-8 <= dividers(k))) {
            valid_lambdas(n_cand) = scaled_lambdas(k) / dividers(k);
            valid_candidates_y(n_cand) =
                candidate_points_in_segment_local(1, k);
            omp_set_lock(&writelock);
            candidate_segments_of_points[orig_ind].push_back(
                std::make_tuple(segment_index, n_cand));
            omp_unset_lock(&writelock);
            valid_candidate_points_in_segment.middleCols(n_cand, 1) =
                candidate_points_in_segment.middleCols(k, 1);
            n_cand++;
          }
        }
        k++;
      }
      // for all candidates, compute curvilinear coordinates
      if (n_cand) {
        Eigen::Matrix2Xd candidates_point_in =
            valid_candidate_points_in_segment.block(0, 0, 2, n_cand);
        Eigen::RowVectorXd lambdas = valid_lambdas.block(0, 0, 1, n_cand);
        Eigen::RowVectorXd p_local_y =
            valid_candidates_y.block(0, 0, 1, n_cand);
        // compute signed pseudo distance
        Eigen::RowVectorXd pseudo_distance =
            segment->computePseudoDistance(lambdas, candidates_point_in);
        // compute signed pseudo distance
        Eigen::RowVectorXd signed_distance(lambdas.cols());
        for (int i = 0; i < lambdas.cols(); i++) {
          if (std::isless(p_local_y(i), 0.)) {
            signed_distance(i) = -pseudo_distance(i);
          } else {
            signed_distance(i) = pseudo_distance(i);
          }
        }
        // compute longitudinal coordinate
        Eigen::RowVectorXd lengths =
            Eigen::RowVectorXd::Constant(n_cand, segment->length_);
        Eigen::RowVectorXd s_offset = Eigen::RowVectorXd::Constant(
            n_cand, this->segmentsLongitudinalCoordinates()[segment_index]);
        Eigen::RowVectorXd lon_coord = lengths.cwiseProduct(lambdas) + s_offset;

        s_results[segment_index] = lon_coord.eval();
        l_results[segment_index] = signed_distance.eval();
      }
    }
  }

  // segment index, longitudinal coordinate, lateral coordinate
  std::vector<std::vector<std::tuple<int, double, double>>>
      groups_of_curvil_points(groups_of_points.size());
  for (int i = 0; i < groups_of_points.size(); i++) {
    groups_of_curvil_points[i].resize(groups_of_points[i].size());
  }
  this->determineCurvilinearCoordinatesAndSegmentIdx(
      candidate_segments_of_points, s_results, l_results, num_omp_threads,
      groups_of_curvil_points);
  omp_destroy_lock(&writelock);
  return groups_of_curvil_points;
}


#if ENABLE_SERIALIZER

int CurvilinearCoordinateSystem::serialize(std::ostream &output_stream) const {
  return serialize::serialize(*this, output_stream);
}

CurvilinearCoordinateSystemConstPtr CurvilinearCoordinateSystem::deserialize(
    std::istream &input_stream) {
  CurvilinearCoordinateSystemConstPtr ret;
  if (!serialize::deserialize(ret, input_stream)) {
    return ret;
  } else
    return CurvilinearCoordinateSystemConstPtr(0);
}

namespace serialize {
ICurvilinearCoordinateSystemExport *exportObject(
    const geometry::CurvilinearCoordinateSystem &);
}

serialize::ICurvilinearCoordinateSystemExport *
CurvilinearCoordinateSystem::exportThis(void) const {
  return serialize::exportObject(*this);
}
#endif

}  // namespace geometry
