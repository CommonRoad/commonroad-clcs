#include "geometry/path_segments.h"

#include "geometry/clcs_exceptions.h"

namespace geometry {

// constructor
PathSegments::PathSegments(const EigenPolyline& reference_path_input,
                           const double eps2) :
                           eps2_(eps2)
{
    // initialize attributes
    this->segment_longitudinal_coord_.push_back(0.0);

    // create sequence of segments representing the reference path from the input polyline
    this->createSegmentsAndReferencePath(reference_path_input);

    // set number of segments
    this->num_segments_ = this->segment_list_.size();

    // set number of points in reference path polyline
    this->num_path_points_ = this->reference_path_.size();
}

std::size_t PathSegments::getNumSegments() const {
    return this->num_segments_;
}

std::size_t PathSegments::getNumPathPoints() const {
    return this->num_path_points_;
}

double PathSegments::getLength() const {
    return this->length_;
}

const EigenPolyline &PathSegments::getReferencePath() const {
    return this->reference_path_;
}

const std::vector<std::unique_ptr<Segment>> &PathSegments::getSegmentList() const {
    return this->segment_list_;
}

const std::vector<double> &PathSegments::getSegmentLongitudinalCoord() const {
    return this->segment_longitudinal_coord_;
}

const std::unique_ptr<Segment> &PathSegments::getSegmentByID(int idx) const {
    return this->segment_list_[idx];
}

Eigen::Vector2d PathSegments::normal(const double s) const {
    int idx = this->findSegmentIndex(s);
    auto &segment = this->segment_list_[idx];
    return segment->normal(s - this->segment_longitudinal_coord_[idx]);
}

Eigen::Vector2d PathSegments::tangent(const double s) const {
    int idx = this->findSegmentIndex(s);
    auto &segment = this->segment_list_[idx];
    return segment->tangent(s - this->segment_longitudinal_coord_[idx]);
}

void PathSegments::createSegmentsAndReferencePath(const EigenPolyline &ref_path_input) {

    if (this->eps2_ != 0) {
        // extend ref path (if eps2_ != 0) by small segments at start and end
        Eigen::Vector2d tangent1 =
                (ref_path_input[0] - ref_path_input[1]).normalized();

        // create 3 new points at the start of the input path
        Eigen::Vector2d new_point0 = ref_path_input[0] + tangent1 * 3 * this->eps2_;
        Eigen::Vector2d new_point1 = ref_path_input[0] + tangent1 * 2 * this->eps2_;
        Eigen::Vector2d new_point2 = ref_path_input[0] + tangent1 * this->eps2_;

        Eigen::Vector2d tangent2 =
                (ref_path_input.back() - ref_path_input[ref_path_input.size() - 2])
                        .normalized();

        // create 3 new points at the end of the input path
        Eigen::Vector2d new_point3 = ref_path_input.back() + tangent2 * this->eps2_;
        Eigen::Vector2d new_point4 = ref_path_input.back() + tangent2 * 2 * this->eps2_;

        // insert 3 new points at the start
        reference_path_.push_back(new_point0);
        reference_path_.push_back(new_point1);
        reference_path_.push_back(new_point2);
        // insert all points from input path
        reference_path_.insert(reference_path_.end(), ref_path_input.begin(),
                               ref_path_input.end());
        // insert 2 new points at the end
        reference_path_.push_back(new_point3);
        reference_path_.push_back(new_point4);
    } else {
        // don't extend reference path: insert all points from input path
        reference_path_.insert(reference_path_.end(), ref_path_input.begin(),
                               ref_path_input.end());
    }

    // get total size of reference path
    std::size_t ref_path_size{reference_path_.size()};

    // create reference path segments
    this->createSegment(reference_path_[0], reference_path_[1],
                        reference_path_[1] - reference_path_[0],
                        reference_path_[2] - reference_path_[0]);

    for (int i = 1; i < ref_path_size - 2; i++) {
        this->createSegment(reference_path_[i], reference_path_[i + 1],
                            reference_path_[i + 1] - reference_path_[i - 1],
                            reference_path_[i + 2] - reference_path_[i]);
    }
    int last_idx = ref_path_size - 1;
    this->createSegment(reference_path_[last_idx - 1], reference_path_[last_idx],
                        reference_path_[last_idx] - reference_path_[last_idx - 2],
                        reference_path_[last_idx] - reference_path_[last_idx - 1]);

}

void PathSegments::createSegment(const Eigen::Vector2d &pt_1, const Eigen::Vector2d &pt_2,
                                 const Eigen::Vector2d &t_1,  const Eigen::Vector2d &t_2) {
    this->segment_list_.push_back(
            std::make_unique<Segment>(pt_1, pt_2, t_1, t_2));
    this->length_ = this->length_ + this->segment_list_.back()->length();
    this->segment_longitudinal_coord_.push_back(length_);
}

const std::unique_ptr<Segment> &PathSegments::findSegment(double s) const {
    int idx = this->findSegmentIndex(s);
    auto &segment = this->segment_list_[idx];
    return segment;
}

int PathSegments::findSegmentIndex(double s) const {
    try {
        return this->tryFindSegmentIndex(s).value();
    } catch (const std::bad_optional_access& e) {
        throw CurvilinearProjectionDomainLongitudinalError();
    }
}

std::optional<int> PathSegments::tryFindSegmentIndex(double s) const {
    auto idx_fast = this->findSegmentIndex_Fast(s);

#ifndef NDEBUG
    auto idx_slow = this->findSegmentIndex_Slow(s);
    assert(idx_slow == idx_fast);
#endif

    return idx_fast;
}

std::optional<int> PathSegments::findSegmentIndex_Fast(double s) const {
    if ((s < 0) || (s > this->length_)) {
        return std::nullopt;
    }

#ifndef NDEBUG
    bool sorted = std::is_sorted(this->segment_longitudinal_coord_.cbegin(), this->segment_longitudinal_coord_.cend());
  assert(sorted);
#endif

    assert(this->segment_longitudinal_coord_.size() == this->segment_list_.size() + 1);

    // std::lower_bound finds the first segment with a longitudinal coordinate greater or equal s
    // However, we need the last segment with a longitudinal coordinate less than s
    // Therefore, a few special cases need to be handled below
    auto it = std::lower_bound(this->segment_longitudinal_coord_.cbegin(), this->segment_longitudinal_coord_.cend(), s);

    // If lower_bound returns the first segment, then s is lower than all segment longitudinal coordinates
    // and thus outside the projection domain
    if (it == this->segment_longitudinal_coord_.cbegin()) {
        return std::nullopt;
    }

    // Move the iterator to the last segment with a longitudinal coordinate less than s
    it = std::prev(it);

#ifndef NDEBUG
    assert(s >= *it);
  if (std::next(it) != this->segment_longitudinal_coord_.cend()) {
    assert(*it <= *std::next(it));
  }
#endif

    return std::distance(this->segment_longitudinal_coord_.cbegin(), it);
}

std::optional<int> PathSegments::findSegmentIndex_Slow(double s) const {
    if ((s < 0) || (s > this->length_)) {
        return std::nullopt;
    }

    std::optional<int> idx = std::nullopt;
    for (int i = 0; i < this->segment_list_.size(); i++) {
        double s_1 = this->segment_longitudinal_coord_[i];
        double s_2 = this->segment_longitudinal_coord_[i + 1];
        if (std::islessequal(s_1, s) && std::islessequal(s, s_2)) {
            idx = i;
            break;
        }
    }
    return idx;
}

void PathSegments::computeBestProjectionAxisForSegments(const EigenPolyline &upperProjectionDomainBorder,
                                                        const EigenPolyline &lowerProjectionDomainBorder) {
    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    // length of border arrays
    const std::size_t len_upper{upperProjectionDomainBorder.size()};
    const std::size_t len_lower{lowerProjectionDomainBorder.size()};

    for (int i = 0; i < this->segment_list_.size(); i++) {
        auto &segment = this->segment_list_[i];
        // find intersection points of segment normals n_1_ and n_2_ with projection domain border
        // these can be directly retrieved from the upper and lower projection domain border
        EigenPolyline pt(4);
        if (i == 0 || i == this->segment_list_.size() - 1) {
            // first and last segment are not within projection domain
            pt[0] = upperProjectionDomainBorder[len_upper - 1];
            pt[1] = upperProjectionDomainBorder[0];  // end of point 1
            pt[2] = lowerProjectionDomainBorder[len_lower - 1];
            pt[3] = lowerProjectionDomainBorder[0];
        } else if (i >= 1) {
            // for all segments, get corresponding distance to projection domain
            pt[0] = upperProjectionDomainBorder[i - 1];
            pt[1] = upperProjectionDomainBorder[i];
            pt[2] = lowerProjectionDomainBorder[i - 1];
            pt[3] = lowerProjectionDomainBorder[i];
        }

        // find minimum and maximum longitudinal and lateral coordinates when
        // projecting current segment onto the longitudinal and lateral axis
        // (axis-aligned bounding box of current segment)
        double min_segm_x = std::min(segment->pt_1().x(), segment->pt_2().x());
        double max_segm_x = std::max(segment->pt_1().x(), segment->pt_2().x());
        double min_segm_y = std::min(segment->pt_1().y(), segment->pt_2().y());
        double max_segm_y = std::max(segment->pt_1().y(), segment->pt_2().y());

        for (auto p : pt) {
            if (p.x() < min_segm_x) min_segm_x = p.x();
            if (p.x() > max_segm_x) max_segm_x = p.x();

            if (p.y() < min_segm_y) min_segm_y = p.y();
            if (p.y() > max_segm_y) max_segm_y = p.y();
        }

        std::deque<double> min_segm_axis(4);
        std::deque<double> max_segm_axis(4);
        min_segm_axis[0] = min_segm_x - 1e-8;
        max_segm_axis[0] = max_segm_x + 1e-8;
        min_segm_axis[1] = min_segm_y - 1e-8;
        max_segm_axis[1] = max_segm_y + 1e-8;

        // find minimum and maximum longitudinal and lateral coordinates when
        // projecting current segment onto two diagonal axes (45 degrees) (find
        // orientated bounding box of current segment)
        // initialize rotation matrix for 45 degrees rotation (w.r.t. global CoSys)
        Eigen::Matrix2d new_axes;
        new_axes << sqrt(2) / 2, sqrt(2) / 2, -sqrt(2) / 2, sqrt(2) / 2;

        Eigen::Vector2d proj_pt1 = util::projectOntoAxes(new_axes, segment->pt_1());
        Eigen::Vector2d proj_pt2 = util::projectOntoAxes(new_axes, segment->pt_2());

        min_segm_x = std::min(proj_pt1.x(), proj_pt2.x());
        max_segm_x = std::max(proj_pt1.x(), proj_pt2.x());
        min_segm_y = std::min(proj_pt1.y(), proj_pt2.y());
        max_segm_y = std::max(proj_pt1.y(), proj_pt2.y());

        for (auto p : pt) {
            Eigen::Vector2d proj_p = util::projectOntoAxes(new_axes, p);
            if (proj_p.x() < min_segm_x) min_segm_x = proj_p.x();
            if (proj_p.x() > max_segm_x) max_segm_x = proj_p.x();

            if (proj_p.y() < min_segm_y) min_segm_y = proj_p.y();
            if (proj_p.y() > max_segm_y) max_segm_y = proj_p.y();
        }

        min_segm_axis[2] = min_segm_x - 1e-8;
        max_segm_axis[2] = max_segm_x + 1e-8;
        min_segm_axis[3] = min_segm_y - 1e-8;
        max_segm_axis[3] = max_segm_y + 1e-8;

        // find axes where the distance to the road boundary within current segment
        // is minimal
        std::vector<double> diff_vec(4);
        diff_vec[0] = (max_segm_axis[0] - min_segm_axis[0]);
        diff_vec[1] = (max_segm_axis[1] - min_segm_axis[1]);
        diff_vec[2] = (max_segm_axis[2] - min_segm_axis[2]);
        diff_vec[3] = (max_segm_axis[3] - min_segm_axis[3]);

        int best_segm_axis_id =
                std::min_element(diff_vec.begin(), diff_vec.end()) - diff_vec.begin();
        this->best_segm_axis_.push_back(ProjectionAxis(best_segm_axis_id));
        this->min_best_segm_axis_.push_back(min_segm_axis[best_segm_axis_id]);
        this->max_best_segm_axis_.push_back(max_segm_axis[best_segm_axis_id]);
    }

    // End timer and print execution time to console
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();
    auto logger = CLCSLogger::getLogger();
    logger->debug("<computeBestProjectionAxisForSegments()> "
                  "Pre-computing projection axis for segments took {} nanoseconds", duration);
}

std::vector<int> PathSegments::findCandidatePointsInSegment(
        int segment_idx,
        const std::vector<std::pair<double, int>> &pair_projected_coord_and_id) const {
    // get min and max val from precomputed vectors
    double val_min = this->min_best_segm_axis_[segment_idx];
    double val_max = this->max_best_segm_axis_[segment_idx];

    auto low = std::lower_bound(
            pair_projected_coord_and_id.begin(), pair_projected_coord_and_id.end(),
            val_min, [](const std::pair<double, int> &lhs, const double rhs) -> bool {
                return lhs.first < rhs;
            });

    auto high = std::lower_bound(
            pair_projected_coord_and_id.begin(), pair_projected_coord_and_id.end(),
            val_max, [](std::pair<double, int> lhs, const double rhs) -> bool {
                return lhs.first < rhs;
            });
    std::vector<int> candidates_indices;
    for (auto it = low; it < high; it++) {
        int point_id =
                pair_projected_coord_and_id[it - pair_projected_coord_and_id.begin()]
                        .second;
        candidates_indices.push_back(point_id);
    }
    return candidates_indices;
}

std::vector<int> PathSegments::computeCandidateIndices(int segment_idx,
                                                       const std::vector<std::pair<double, int>> &pairs_in_x,
                                                       const std::vector<std::pair<double, int>> &pairs_in_x_rotated,
                                                       const std::vector<std::pair<double, int>> &pairs_in_y,
                                                       const std::vector<std::pair<double, int>> &pairs_in_y_rotated)
                                                       const {
    std::vector<int> candidates_indices;
    if (this->best_segm_axis_[segment_idx] == ProjectionAxis::X_AXIS) {
        candidates_indices = this->findCandidatePointsInSegment(segment_idx, pairs_in_x);
    } else if (this->best_segm_axis_[segment_idx] ==
               ProjectionAxis::X_AXIS_ROTATED) {
        candidates_indices = this->findCandidatePointsInSegment(segment_idx, pairs_in_x_rotated);
    } else if (this->best_segm_axis_[segment_idx] ==
               ProjectionAxis::Y_AXIS) {
        candidates_indices = this->findCandidatePointsInSegment(segment_idx, pairs_in_y);
    } else if (this->best_segm_axis_[segment_idx] ==
               ProjectionAxis::Y_AXIS_ROTATED) {
        candidates_indices = this->findCandidatePointsInSegment(segment_idx, pairs_in_y_rotated);
    }

    return candidates_indices;
}

std::list<int> PathSegments::determineIndicesRange(
        int segment_idx, int previous_segment_idx) const {
    std::list<int> indices_range;
    if (segment_idx < previous_segment_idx) {
        indices_range.resize(previous_segment_idx - segment_idx + 1);
        std::iota(indices_range.rbegin(), indices_range.rend(), segment_idx);
    } else {
        indices_range.resize(segment_idx - previous_segment_idx + 1);
        std::iota(indices_range.begin(), indices_range.end(), previous_segment_idx);
    }
    return indices_range;
}

void PathSegments::interpolatePointsBetweenSegments(
        const Eigen::Vector2d cartesian_point,
        const Eigen::Vector2d next_cartesian_point,
        const std::list<int> indices_range,
        const double normal_length,
        EigenPolyline &transformed_polygon) const {
    Eigen::Vector2d first_curvilinear_point = transformed_polygon.back();
    EigenPolyline tmp_vertices;
    tmp_vertices.push_back(cartesian_point);
    for (const auto &k : indices_range) {
        Eigen::Vector2d intersection_point;
        const auto &segment = this->getSegmentByID(k);
        // compute intersection point between segment normal and
        // straight line between cartesian_point and next_cartesian_point
        bool intersect = geometry::util::intersectionSegmentSegment(
                segment->pt_2() - (normal_length + 10.0) *
                                  segment->normalSegmentEnd(),
                segment->pt_2() + (normal_length + 10.0) *
                                  segment->normalSegmentEnd(),
                next_cartesian_point, tmp_vertices.back(), intersection_point);

        if (intersect) {
            tmp_vertices.push_back(intersection_point);
            Eigen::Vector2d curvilinear_point = segment->convertToCurvilinearCoords(
                    intersection_point[0], intersection_point[1]);
            curvilinear_point +=
                    Eigen::Vector2d(this->segment_longitudinal_coord_[k], 0);

            // in the case that the last transformed point lies near the new segment,
            // numerically errors may occur when constructing a polygon, since the new
            // curvilinear_point and the last curvilinear point are almost the same.
            // We skip that point and do not add it to the coordinate list
            if (k == indices_range.front() &&
                first_curvilinear_point.isApprox(curvilinear_point, 10e-8)) {
                transformed_polygon.pop_back();
                transformed_polygon.push_back(curvilinear_point);
            } else {
                transformed_polygon.push_back(curvilinear_point);
            }
        }
    }
}

} // namespace geometry