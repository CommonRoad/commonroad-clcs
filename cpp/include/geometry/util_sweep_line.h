/**
 * @file    util_sweep_line.h
 * @brief   Bentley-Ottmann Sweep-Line algorithm
 * @author  Evald Nexhipi
 */

#ifndef SWEEP_LINE_UTIL_H
#define SWEEP_LINE_UTIL_H

#include <cstdlib>
#include <cmath>
#include <queue>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include "geometry/util.h"


namespace geometry {

namespace sweep_line_util {


/**
 * precision constant epsilon used within sweep line functions
 */
namespace Constants
{
    const double EPSILON = ldexp(1.0, -36);
}


/**
 * Enum to define status of a point
 * STATUS_UNDEFINED
 * STATUS_LEFT: Starting point of a Segment
 * STATUS_RIGHT: Ending point of a Segment
 * STATUS_INTERSECTION: Intersection point (i.e., point lies somewhere in the center, between start and end point of a segment)
 */
enum PointStatus
{
    STATUS_UNDEFINED,
    STATUS_LEFT,
    STATUS_RIGHT,
    STATUS_INTERSECTION
};


/**
 * Structure for Point within Sweep Line and attributes of the point
 * A point is defined by its x and y coordinates.
 */
struct Point
{
public:
    // constructor of the struct
    Point(const Eigen::Vector2d& coordinates);
    // overloaded "<" operator:
    bool operator<(const Point &p) const;

    // getters for coordinates
    Eigen::Vector2d coordinates() const {return coordinates_;}
    double x() const {return coordinates_.x();}
    double y() const {return coordinates_.y();}

    // make intersection point
    void CreateIntersectionPoint(const int segMain, const int segOther, const int idx);

    // status of point
    PointStatus status;
    // IDs of segments of the original polyline belonging to the point
    int seg_main;
    int seg_other;
    // unique id of point
    int id;

private:
    // x and y coordinate of a point
    Eigen::Vector2d coordinates_;
};

/**
 * Structure for a SegmentLine
 * A Segment Line is defined by it's left and right point.
 * Additional attributes are the slope of the segment and it's y intercept (this defines the standard line equation)
 */
struct SegmentLine
{
public:
    // constructor of the struct
    SegmentLine(const Eigen::Vector2d& coords_left, const Eigen::Vector2d& coords_right, const int idxSegMain) :
        pt_left(Point(coords_left)),
        pt_right(Point(coords_right))
    {
        this->pt_left.status = STATUS_LEFT;
        this->pt_left.seg_main = idxSegMain;
        this->pt_right.status = STATUS_RIGHT;
        this->pt_right.seg_main = idxSegMain;
        this->ComputeSlopeIntercept();
    }

    // left and right point of the SegmentLine
    Point pt_left;
    Point pt_right;

    double ComputeYatX(const double x) const;

private:
    // slope and intercept (line equation)
    double slope_;
    double intercept_;

    void ComputeSlopeIntercept();
};


/**
 * Structure to store intersections and avoid duplicate intersections: if a point is already there, it is not added
 * @param m_pts [std::map]: Map to store an Intersection point
 */
struct Intersections
{
public:
    // add an intersection to map_segments_to_point_
    void Add(const Point &point);

    // creates map of segments to pair <intersecting segment, intersection point>
    std::unordered_map<int, std::vector<std::pair<int, Eigen::Vector2d>>> getMapSegmentToSegment();

private:
    // maps pair of intersecting segment IDs <int, int> to intersection point <Eigen::Vector2d>
    std::map<std::pair<int, int>, Eigen::Vector2d> map_segment_pairs_to_point_;
};


/**
 * Structure for an entry of the Sweep Line.
 * An entry is defined by the y coordinate (vertical sweep line) and the segment ID
 */
struct SweepLineEntry
{
    double y_coord;
    int seg_id;

    // overloaded "<" operator, required for sorting a vector of SweepLineEntries
    bool operator<(const SweepLineEntry &entry) const;
};

/**
 * Main Structure for the Sweep Line
 */
struct SweepLine
{
public:
    // vector of segments to check for intersections
    const std::vector<SegmentLine> *vec_segments;

    // get entry for a given segment number
    int FindEntryBySegID(const int seg_id) const;
    // insert an entry in vec_entries
    void InsertEntry(const SweepLineEntry &entry, const double x);
    // remove and entry from vec_entries at a given index
    void RemoveEntryAtPos(const int pos);
    // update and sort entries according to ascending y_coord
    void UpdateAndSortEntries(const double x);

    // return successor/predecessor segment number for a given index
    int GetSuccessorAtPos(const int pos) const;
    int GetPredecessorAtPos(const int pos) const;

private:
    // vector of SweepLineEntries
    std::vector<SweepLineEntry> vec_entries_;
};

/**
 * Function which checks for an intersection between two SegmentLines
 * @param &a: first SegmentLine
 * @param &b: second SegmentLine
 * @param &intersection_point: intersection point (in case Segments intersect)
 */
static bool CheckIntersection(const SegmentLine &seg_a, const SegmentLine &seg_b, Eigen::Vector2d& intersection_point);

/**
 * Function determines intersections between a set of line segments via a vertical sweep line approach
 * based on a modified version of the Bentley-Ottmann algorithm
 */
void SweepLineIntersections(const std::vector<SegmentLine> &segs, Intersections &intersections);

}  // namespace sweep_line_util

}  // namespace geometry

#endif  // SWEEP_LINE_UTIL_H
