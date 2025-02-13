#include "geometry/projection_domain.h"

namespace geometry{

// constructor
ProjectionDomain::ProjectionDomain(const std::vector<std::unique_ptr<Segment>> &seg_list,
                                   const std::vector<double> curv_vec,
                                   const std::vector<double> curv_rad_vec,
                                   const std::vector<double> seg_lon_coord_vec,
                                   const double length, const double def_proj_domain_lim,
                                   const double eps, const int method) :
                                   curvature_vec_(curv_vec),
                                   curvature_radius_vec_(curv_rad_vec),
                                   segment_lon_coord_(seg_lon_coord_vec),
                                   length_(length),
                                   default_projection_domain_limit_(def_proj_domain_lim),
                                   eps_(eps),
                                   proj_domain_method_(method)
{
    // check valid method
    if(method != 1 && method != 2) {
        throw InvalidMethodError();
    }

    // approximate projection domain
    this->approximateProjectionDomain(seg_list);
}

EigenPolyline ProjectionDomain::projectionDomainBorder() const {
    return this->projection_domain_border_;
}

EigenPolyline ProjectionDomain::curvilinearProjectionDomainBorder() const {
    return this->curvilinear_projection_domain_border_;
}

EigenPolyline ProjectionDomain::upperProjectionDomainBorder() const {
    return this->upper_projection_domain_border_;
}

EigenPolyline ProjectionDomain::lowerProjectionDomainBorder() const {
    return this->lower_projection_domain_border_;
}

bool ProjectionDomain::cartesianPointInProjectionDomain(double x, double y) const {
    return boost::geometry::covered_by(point_type(x, y), this->projection_domain_);
}

std::tuple<bool, bool> ProjectionDomain::curvilinearPointInProjectionDomain(const std::unique_ptr<Segment>& segment_ptr,
                                                                            int seg_idx,
                                                                            const double seg_lon_coord,
                                                                            double s,
                                                                            double l) const {
    bool in_longitudinal_bounds = true;
    bool in_lateral_bounds = true;

    // check if longitudinal coordinate is outside of longitudinal bounds of reference path
    if ( (s < 0) || (s > this->length_) || (seg_idx < 0) ) {
        in_longitudinal_bounds = false;
        in_lateral_bounds = false;

    } else {
        // check if lateral coordinate is within lateral range of given seg_idx
        // convert upper and lower projection domain point to segment
        // TODO: precompute curvilinear points for each segment already at initialization in approximateCurvilinearProjectionDomain
        EigenPolyline pt(4);
        pt[0] = this->upper_projection_domain_border_[seg_idx - 1];
        pt[1] = this->upper_projection_domain_border_[seg_idx];
        pt[2] = this->lower_projection_domain_border_[seg_idx - 1];
        pt[3] = this->lower_projection_domain_border_[seg_idx];
        double lambda{0.0};
        EigenPolyline curvilinear_points(4);
        for (int i = 0; i < 4; i++) {
            Eigen::Vector2d curv_pt = segment_ptr->convertToCurvilinearCoords(pt[i].x(), pt[i].y(), lambda);
            curvilinear_points[i] = curv_pt;
        }
        // get lateral interval
        double lambda_s = segment_ptr->computeLambda(s - seg_lon_coord);
        double l_max = curvilinear_points[0].y() + (curvilinear_points[1].y() - curvilinear_points[0].y()) * lambda_s;
        double l_min = curvilinear_points[2].y() + (curvilinear_points[3].y() - curvilinear_points[2].y()) * lambda_s;

        // check coordinate in interval
        if ((l < l_min) || (l > l_max)) {
            in_lateral_bounds = false;
        }
    }
    return std::make_tuple(in_longitudinal_bounds, in_lateral_bounds);
}

void ProjectionDomain::approximateCurvilinearProjectionDomain(const std::vector<std::unique_ptr<Segment>>& segment_list,
                                                              const std::vector<double>& vec_leftSegmentDistances,
                                                              const std::vector<double>& vec_rightSegmentDistances) {
    // populate curvilinear projection domain border polyline with computed distances
    Eigen::Vector2d tmp_pt(0.0, 0.0);

    // left distances
    for(int i = 0; i < vec_leftSegmentDistances.size() - 1; i++) {
        tmp_pt[0] = this->segment_lon_coord_[i] + segment_list[i]->length();     // s coordinate
        tmp_pt[1] = vec_leftSegmentDistances[i];                                 // d coordinate

        // add to curvilinear border
        this->curvilinear_projection_domain_border_.push_back(tmp_pt);
    }

    // right distances (iterate in reverse)
    for(auto iter = vec_rightSegmentDistances.end() - 2; iter >= vec_rightSegmentDistances.begin(); iter--){
        std::size_t idx = iter - vec_rightSegmentDistances.begin();
        tmp_pt[0] = this->segment_lon_coord_[idx] + segment_list[idx]->length();
        tmp_pt[1] = *iter * (-1);

        // add to curvilinear border
        this->curvilinear_projection_domain_border_.push_back(tmp_pt);
    }

    // to construct Polygon, the vertices must be closed. Add point at the end such that first and last vertex coincide
    if(!this->curvilinear_projection_domain_border_.front().isApprox(
            this->curvilinear_projection_domain_border_.back())) {
        this->curvilinear_projection_domain_border_.push_back(this->curvilinear_projection_domain_border_.front());
    }

    // create Boost polygon from border polyline
    util_proj::polylineToBoostPolygon(this->curvilinear_projection_domain_border_,
                                      this->curvilinear_projection_domain_);
}

void ProjectionDomain::approximateProjectionDomain(const std::vector<std::unique_ptr<Segment>>& segment_list) {
    // get logger
    auto logger = CLCSLogger::getLogger();
    logger->debug("<approximateProjectionDomain()> Using computation method: {}", this->proj_domain_method_);

    // number of segments
    const std::size_t num_segments{segment_list.size()};

    // initialize distance vectors
    std::vector<double> vec_left_distances(num_segments, this->default_projection_domain_limit_);
    std::vector<double> vec_right_distances(num_segments, this->default_projection_domain_limit_);

    // Start timer
    auto startTime = std::chrono::high_resolution_clock::now();

    if (this->proj_domain_method_ == 1) {
        // METHOD 1: use worst-case (largest) curvature of reference path to compute projection domain limit
        double min_radius, max_radius;
        std::tie(min_radius, max_radius) = this->computeProjectionDomainLimits(segment_list);

        // fill vectors with computed limits. Left: positive (max_radius). Right: negative (-min_radius)
        std::fill(vec_left_distances.begin(), vec_left_distances.end(), max_radius);
        std::fill(vec_right_distances.begin(), vec_right_distances.end(), -min_radius);

    } else if (this->proj_domain_method_ == 2) {
        // METHOD 2: use sweep-line to tightly under-approximate the projection domain
        // compute distances of projection domain border vertices
        this->computeProjectionDomainExact(segment_list, vec_left_distances, vec_right_distances);
    }

    // compute polyline of projection domain border
    this->projection_domain_border_ = this->computeProjectionDomainBorder(segment_list,
                                                                          vec_left_distances,
                                                                          vec_right_distances);

    // End timer and print execution time to console
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime).count();

    // log to console
    logger->debug("<approximateProjectionDomain()> Computing projection domain took: {} nanoseconds", duration);

    // create boost polygon for Cartesian projection domain from border polyline
    util_proj::polylineToBoostPolygon(this->projection_domain_border_, this->projection_domain_);

    // approximate Curvilinear projection domain
    this->approximateCurvilinearProjectionDomain(segment_list,
                                                 vec_left_distances,
                                                 vec_right_distances);
}

std::tuple<double, double> ProjectionDomain::computeProjectionDomainLimits(
        const std::vector<std::unique_ptr<Segment>>& segment_list) const {

    // vector of intersection distances
    std::vector<double> intersection_distances;
    // compute intersections between lateral segment vectors
    for (int i = 0; i < segment_list.size() - 1; i++) {
        const auto &segment_1 = segment_list[i];
        // get distance of normal from curvature radius at segment
        double curv = this->curvature_vec_[i + 1];
        double curv_rad = this->curvature_radius_vec_[i + 1];    // pt2 of segment i
        double dist_pos;
        double dist_neg;
        if (curv >= 0.0) {
            dist_pos = std::min(curv_rad, this->default_projection_domain_limit_) + this->eps_;
            dist_neg = this->default_projection_domain_limit_ + this->eps_;
        } else {
            dist_pos = this->default_projection_domain_limit_ + this->eps_;
            dist_neg = std::min(curv_rad, this->default_projection_domain_limit_) + this->eps_;
        }

        Eigen::Vector2d p_1 =
                segment_1->pt_2() - dist_neg * segment_1->normalSegmentEnd();
        Eigen::Vector2d p_2 =
                segment_1->pt_2() + dist_pos * segment_1->normalSegmentEnd();

        // get distance of normal from curvature radius at segment
        curv = this->curvature_vec_[i + 2];
        curv_rad = this->curvature_radius_vec_[i + 2];    // pt2 of segment i + 1
        if (curv >= 0.0) {
            dist_pos = std::min(curv_rad, this->default_projection_domain_limit_) + this->eps_;
            dist_neg = this->default_projection_domain_limit_ + this->eps_;
        } else {
            dist_pos = this->default_projection_domain_limit_ + this->eps_;
            dist_neg = std::min(curv_rad, this->default_projection_domain_limit_) + this->eps_;
        }

        const auto &segment_2 = segment_list[i + 1];
        Eigen::Vector2d p_3 =
                segment_2->pt_2() - dist_neg * segment_2->normalSegmentEnd();
        Eigen::Vector2d p_4 =
                segment_2->pt_2() + dist_pos * segment_2->normalSegmentEnd();

        Eigen::Vector2d intersection_point(0., 0.);
        bool intersect = geometry::util::intersectionLineLine(p_1, p_2, p_3, p_4,
                                                              intersection_point);
        if (intersect) {
            Eigen::Vector2d d_1 = intersection_point - segment_1->pt_2();
            Eigen::Vector2d d_2 = intersection_point - segment_2->pt_2();
            double dot_d_1 = (segment_1->normalSegmentEnd())
                    .dot(intersection_point - segment_1->pt_2());
            intersection_distances.push_back(copysign(1.0, dot_d_1) * d_1.norm());
        }
    }

    // get the smallest positive distance and the greatest negative distance
    double min_radius = -this->default_projection_domain_limit_;
    double max_radius = this->default_projection_domain_limit_;
    for (const auto &dis : intersection_distances) {
        if (dis > 0 && dis < max_radius) {
            max_radius = (dis - this->eps_ > 0) ? dis - this->eps_ : dis;
        } else if (dis < 0 && dis > min_radius) {
            min_radius = (dis + this->eps_ < 0) ? dis + this->eps_ : dis;
        }
    }
    return std::make_tuple(min_radius, max_radius);
}

EigenPolyline ProjectionDomain::computeProjectionDomainBorder(const std::vector<std::unique_ptr<Segment>>& segment_list,
                                                              const std::vector<double>& vec_leftSegmentDistances,
                                                              const std::vector<double>& vec_rightSegmentDistances) {
    // create upper (positive sign) and lower (negative sign) border
    // The distance is added to each segment end point along the normal, and we get the right and left
    // projection domain points, which are then connected.
    EigenPolyline projection_domain_border;
    // iterate over all segments
    for (int i = 0; i < segment_list.size() - 1; i++) {
        const auto &segment = segment_list[i];
        this->upper_projection_domain_border_.push_back(
                segment->pt_2() + vec_leftSegmentDistances[i] * segment->normalSegmentEnd());
        this->lower_projection_domain_border_.push_back(
                segment->pt_2() - vec_rightSegmentDistances[i] * segment->normalSegmentEnd());
    }

    // concatenate projection domain border
    projection_domain_border.insert(projection_domain_border.end(),
                                    this->upper_projection_domain_border_.begin(),
                                    this->upper_projection_domain_border_.end());
    projection_domain_border.insert(projection_domain_border.end(),
                                    this->lower_projection_domain_border_.rbegin(),
                                    this->lower_projection_domain_border_.rend());

    // to construct Polygon, the vertices must be closed. Add point at the end such that first and last vertex coincide
    if (!this->upper_projection_domain_border_.front().isApprox(
            this->lower_projection_domain_border_.front())) {
        projection_domain_border.push_back(
                this->upper_projection_domain_border_.front());
    }
    return projection_domain_border;
}

void ProjectionDomain::computeProjectionDomainExact(const std::vector<std::unique_ptr<Segment>>& segment_list,
                                                    std::vector<double>& vec_leftSegmentDistances,
                                                    std::vector<double>& vec_rightSegmentDistances)
{
    // initialize list of Sweep Line Segments for left and right case
    std::vector<util_sweep::SegmentLine> segmentsLeft;
    std::vector<util_sweep::SegmentLine> segmentsRight;
    int list_size = segment_list.size();
    segmentsLeft.reserve(list_size - 1);
    segmentsRight.reserve(list_size - 1);

    /*
     * Prepare Sweep Line Algorithm
     * Iterate over all Segments of the Polyline
     * retrieve start and end points of segment normal vectors (left and right normal)
     * Create a SegmentLine for the sweep line for each normal and add to list of left or right segment
     */
    for (int i = 0; i < list_size - 1; i++) {
        const auto &segment_ = segment_list[i];

        // normal vectors of Polyline segment are defined always as (-t[1], t[0]) given tangent vector which faces in
        // the direction of the polyline

        // "left" and "right" is thereby defined in direction of the polyline
        // get distance of normal from curvature radius at segment
        double curv = this->curvature_vec_[i + 1];
        double curv_rad = this->curvature_radius_vec_[i + 1];    // pt2 of segment i
        double dist_pos;
        double dist_neg;
        if (curv >= 0.0) {
            dist_pos = std::min(curv_rad, this->default_projection_domain_limit_) + this->eps_;
            dist_neg = this->default_projection_domain_limit_ + this->eps_;
        } else {
            dist_pos = this->default_projection_domain_limit_ + this->eps_;
            dist_neg = std::min(curv_rad, this->default_projection_domain_limit_) + this->eps_;
        }

        Eigen::Vector2d pLeftStart = segment_->pt_2() + dist_pos * segment_->normalSegmentEnd();
        Eigen::Vector2d pLeftEnd = segment_->pt_2();

        Eigen::Vector2d pRightStart = segment_->pt_2();
        Eigen::Vector2d pRightEnd = segment_->pt_2() - dist_neg * segment_->normalSegmentEnd();

        // list of left segments
        if (pLeftStart.x() < pLeftEnd.x()) {
            util_sweep::SegmentLine segment_line(pLeftStart, pLeftEnd, i);
            segmentsLeft.push_back(segment_line);
        } else {
            util_sweep::SegmentLine segment_line(pLeftEnd, pLeftStart, i);
            segmentsLeft.push_back(segment_line);
        }

        // list of right segments
        if (pRightStart.x() < pRightEnd.x()) {
            util_sweep::SegmentLine segment_line(pRightStart, pRightEnd, i);
            segmentsRight.push_back(segment_line);
        } else {
            util_sweep::SegmentLine segment_line(pRightEnd, pRightStart, i);
            segmentsRight.push_back(segment_line);
        }
    }

    // We get two arrays one for the segments that intersect on the left and the other on the right
    // Call Sweep Line for left segments and determine intersections on the left side
    util_sweep::Intersections leftIntersections;
    SweepLineIntersections(segmentsLeft, leftIntersections);
    std::unordered_map<int, std::vector<std::pair<int, Eigen::Vector2d>>> leftIntersectionPairs = leftIntersections.getMapSegmentToSegment();

    // Call Sweep Line for right segments and determine intersections on the right side
    util_sweep::Intersections rightIntersections;
    SweepLineIntersections(segmentsRight, rightIntersections);
    std::unordered_map<int, std::vector<std::pair<int, Eigen::Vector2d>>> rightIntersectionPairs = rightIntersections.getMapSegmentToSegment();

    // Determine proj domain distances for each segment
    this->getSegmentDistances(segment_list, vec_leftSegmentDistances, leftIntersectionPairs, "LEFT");
    this->getSegmentDistances(segment_list, vec_rightSegmentDistances, rightIntersectionPairs, "RIGHT");
}

void ProjectionDomain::getSegmentDistances(const std::vector<std::unique_ptr<Segment>>& segment_list,
                                           std::vector<double>& vec_segment_distances,
                                           std::unordered_map<int, std::vector<std::pair<int, Eigen::Vector2d>>>& map_intersections,
                                           std::string direction) {
    // determine signed distance according to direction
    // "left" and "right" are defined in direction of the polyline
    // the unit normal vector always faces in left direction --> positive sign for "LEFT"
    int sign{0};
    if (direction == "LEFT") {
        sign = 1;
    } else if (direction == "RIGHT") {
        sign = -1;
    }

    // set for keeping track of processed segments
    std::unordered_map<int, std::vector<std::pair<int, double>>> map_intersections_distances;

    //iterate over segments which have an intersection
    for (auto &iter : map_intersections) {
        // get corresponding segment of polyline at index
        int seg_idx = iter.first;
        const auto &segment_i = segment_list[seg_idx];
        // max limit (initialized as default maximum limit)
        double limit = this->default_projection_domain_limit_;

        // iterate over intersecting segments and compute lateral distance
        for (auto &pair_segment_point : iter.second) {
            // get stored intersection point determined from sweep line
            Eigen::Vector2d intersection_point = pair_segment_point.second;

            // compute lateral distance to segment and select minimum distance as limit
            Eigen::Vector2d d_1 = intersection_point - segment_i->pt_2();
            double dot_d_1 = (segment_i->normalSegmentEnd()).dot(intersection_point - segment_i->pt_2());
            double dist = sign * copysign(1.0, dot_d_1) * d_1.norm();

            map_intersections_distances[seg_idx].push_back(std::make_pair(pair_segment_point.first, dist));

            if (dist < limit) {
                limit = dist;
            }
        }
        vec_segment_distances[seg_idx] = limit - this->eps_;
    }
}

std::vector<EigenPolyline> ProjectionDomain::determineSubsetOfPolygonWithinProjectionDomain(
        const EigenPolyline &polygon) const {
    return this->polygonWithinProjectionDomain(polygon, this->projection_domain_);
}

std::vector<EigenPolyline> ProjectionDomain::determineSubsetOfPolygonWithinCurvilinearProjectionDomain(
        const EigenPolyline &polygon) const {
    return this->polygonWithinProjectionDomain(polygon, this->curvilinear_projection_domain_);
}

std::vector<EigenPolyline> ProjectionDomain::polygonWithinProjectionDomain(
        const EigenPolyline &polygon, const polygon_type &projection_domain) const {

    std::vector<EigenPolyline> polygons_within_projection_domain;
    if (polygon.size() == 0) return polygons_within_projection_domain;

    // over-approximate polygon with axis-aligned rectangle
    polygon_type poly_aabb;
    util_proj::overapproximatePolygonAABB(polygon, poly_aabb);

    if (boost::geometry::within(poly_aabb, projection_domain)) {
        polygons_within_projection_domain.push_back(polygon);
    } else {
        polygons_within_projection_domain = util_proj::polygonWithinPolygonBoost(polygon, projection_domain);
    }
    return polygons_within_projection_domain;
}
}  // namespace geometry