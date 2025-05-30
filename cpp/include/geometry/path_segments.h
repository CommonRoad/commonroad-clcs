#ifndef PATH_SEGMENTS_H
#define PATH_SEGMENTS_H

#include <vector>
#include <numeric>
#include <optional>
#include <list>
#include "geometry/segment.h"
#include "geometry/clcs_types.h"
#include "geometry/clcs_logger.h"


namespace geometry {

/**
 * @brief enum class to define ProjectionAxis Types
 */
enum class ProjectionAxis {
    X_AXIS = 0,
    Y_AXIS = 1,
    X_AXIS_ROTATED = 2,
    Y_AXIS_ROTATED = 3
};

class PathSegments {
public:
    /**
     * Constructor of the class for handling the reference path of the CLCS as
     * a sequence of segments. The path segments are constructed from a Polyline
     * representing the reference path.
     *
     * @param reference_path_input polyline representing the reference path
     * @param eps2 [optional] precision parameter, which adds additional segments to the start and end of
     * reference_path_input. Avoids projection errors near the borders of the reference path.
     */
    PathSegments(const EigenPolyline& reference_path_input,
                 const double eps2);

    ~PathSegments() = default;      ///< default destructor

    /**
     * Getter for number of segments
     */
    std::size_t getNumSegments() const;

    /**
     * Getter for number of points in path polyline (num_segments_ + 1)
     */
    std::size_t getNumPathPoints() const;

    /**
     * Getter for total length of path
     */
    double getLength() const;

    /**
     * Getter returning a reference to the (extended) ref path used by the CLCS as a polyline
     */
    const EigenPolyline &getReferencePath() const;

    /**
     * Getter returning a reference to the list (vector) of segments in PathSegments
     */
    const std::vector<std::unique_ptr<Segment>> &getSegmentList() const;

    /**
     * Getter returning reference to vector of longitudinal coordinates of segments in PathSegments
     */
    const std::vector<double> &getSegmentLongitudinalCoord() const;

    /**
     * Getter returning a reference to a segment pointer by ID
     *
     * @param idx index of segment in vector segment_list_
     * @return reference to segment pointer
     */
    const std::unique_ptr<Segment> &getSegmentByID(int idx) const;

    /**
     * Normal vector at a specific longitudinal coordinate
     */
    Eigen::Vector2d normal(const double s) const;

    /**
     * Tangent vector at a specific longitudinal coordinate
     */
    Eigen::Vector2d tangent(const double s) const;

    /**
     * Returns a reference to the pointer of the corresponding segment for a given longitudinal coordinate s
     *
     * @param s longitudinal coordinate
     * @return reference to segment pointer
     */
    const std::unique_ptr<Segment> &findSegment(double s) const;

    /**
     * Returns the corresponding segment index for a given longitudinal coordinate s.
     * Additionally checks if coordinate is within longitudinal bounds of reference path.
     *
     * @param s longitudinal coordinate
     * @return segment index
     */
    int findSegmentIndex(double s) const;

    /**
     * Wrapper method for finding segment index for a given longitudinal coordinate s.
     */
    std::optional<int> tryFindSegmentIndex(double s) const;

    /**
     * Fast method for finding segment index for a given longitudinal coordinate s.
     * Uses binary search O(log n).
     */
    std::optional<int> findSegmentIndex_Fast(double s) const;

    /**
     * Slow method for finding segment index for a given longitudinal coordinate s.
     * Uses naive approach which iterated over all segments.
     */
    std::optional<int> findSegmentIndex_Slow(double s) const;

    /**
     * Computes a projection axis for each segment to quickly determine global
     * points that lie within the segment. Only the points that lie between the
     * minimum and the maximum coordinates projected on the axis are considered to
     * be within the segment.
     *
     * @param upperProjectionDomainBorder Upper projection domain border as polyline
     * @param lowerProjectionDomainBorder Lower projection domain border as polyline
     */
    void computeBestProjectionAxisForSegments(const EigenPolyline &upperProjectionDomainBorder,
                                              const EigenPolyline &lowerProjectionDomainBorder);

    /**
     * Given a list of coordinates projected to the best projection axis
     * of a segment, the function returns the indices of the points which may be
     * inside the segment.
     *
     * @param segment_idx ID of the segment
     * @param pair_projected_coord_and_id projected coordinates to the projection
     * axis of the segment with ID segment_idx + unique ID for each projected coordinate
     * @return IDs of projected coordinates which are candidate that lie inside the segment
     */
    std::vector<int> findCandidatePointsInSegment(
            int segment_idx,
            const std::vector<std::pair<double, int>> &pair_projected_coord_and_id) const;

    /**
     * For a given segment index, the function checks the precomputed projection axis and
     * returns the indices of the points which may lie within the segment.
     *
     * @param segment_idx ID of the segment
     * @param pairs_in_x
     * @param pairs_in_x_rotated
     * @param pairs_in_y
     * @param pairs_in_y_rotated
     */
    std::vector<int> computeCandidateIndices(int segment_idx,
                                             const std::vector<std::pair<double, int>> &pairs_in_x,
                                             const std::vector<std::pair<double, int>> &pairs_in_x_rotated,
                                             const std::vector<std::pair<double, int>> &pairs_in_y,
                                             const std::vector<std::pair<double, int>> &pairs_in_y_rotated) const;

    /**
     * Given two segment indices, the function computes the intermediate indices.
     *
     * @param segment_idx
     * @param previous_segment_idx
     */
    std::list<int> determineIndicesRange(int segment_idx, int previous_segment_idx) const;

    /**
     * Given two Cartesian points (cartesian_point and next_cartesian_point), the
     * function computes the intersection points of the line connecting the two
     * points and the segment normals. TODO: remove transformed_polygon
     *
     * @param[in] cartesian_point
     * @param[in] next_cartesian_point
     * @param[in] indices_range IDs of segments
     * @param[in/out] transformed_polygon transformed intersection points are appended to transformed_polygon
     */
    void interpolatePointsBetweenSegments(
            const Eigen::Vector2d cartesian_point,
            const Eigen::Vector2d next_cartesian_point,
            const std::list<int> indices_range,
            const double normal_length,
            EigenPolyline &transformed_polygon) const;


private:
    /**
     * Creates the sequence of path segments from an input reference path (polyline).
     * Extends original polyline at beginning and end (if desired) by a precision value to avoid projection errors.
     *
     * @param ref_path_input input reference path as a polyline
     */
    void createSegmentsAndReferencePath(const EigenPolyline &ref_path_input);

    /**
     * Creates a new segment of the path segments.
     *
     * @param p_1 start point of segment
     * @param p_2 end point of segment
     * @param t_1 tangent vector at the start of segment
     * @param t_2 tangent vector at the end of segment
     */
    void createSegment(const Eigen::Vector2d &pt_1, const Eigen::Vector2d &pt_2,
                       const Eigen::Vector2d &t_1,  const Eigen::Vector2d &t_2);

    EigenPolyline reference_path_;                            ///< reference path (with extensions) as a Polyline
    std::size_t num_path_points_;                             ///< total number of path points

    std::vector<std::unique_ptr<Segment>> segment_list_;      ///< vector/list of all segments in the reference path
    std::vector<double> segment_longitudinal_coord_;          ///< vector of lon. coordinates of segments
    std::size_t num_segments_;                                ///< total number of segments

    std::deque<ProjectionAxis> best_segm_axis_;               ///< best projection axis of segments
    std::vector<double> min_best_segm_axis_;                  ///< vector of min best projection axis
    std::vector<double> max_best_segm_axis_;                  ///< vector of max best projection axis

    double length_{0.0};                                      ///< total length of reference path

    const double eps2_;                                       ///< extrapolation value for start and end of reference
};

} // namespace geometry

#endif //PATH_SEGMENTS_H
