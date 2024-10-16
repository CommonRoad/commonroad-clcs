#include "geometry/util_sweep_line.h"

namespace geometry {

namespace sweep_line_util {

// Constructor of Point struct
Point::Point(const Eigen::Vector2d& coordinates) :
        coordinates_(coordinates),
        seg_main(-1),
        status(STATUS_UNDEFINED),
        seg_other(-1),
        id(-1)
{
}

//TODO: use logical && operator
bool Point::operator<(const Point &p) const {
  return this->x() > p.x() || (this->x() == p.x() & this->y() > p.y());
}

void Point::Print(void) {
  std::cout
        << "id = " << id
        << " x = " << this->x()
        << " y = " << this->y()
        << " status = " << this->status
        << " segs = " << this->seg_main << " " << seg_other << std::endl;
}

void Point::CreateIntersectionPoint(const int segMain, const int segOther, const int idx) {
    this->status = STATUS_INTERSECTION;
    this->seg_main = segMain;
    this->seg_other = segOther;
    this->id = idx;
}

void SegmentLine::ComputeSlopeIntercept(void) {
    slope_ = (pt_right.y() - pt_left.y()) / (pt_right.x() - pt_left.x());
    intercept_ = -pt_left.x() * slope_ + pt_left.y();
}

double SegmentLine::ComputeYatX(const double x) const {
    return slope_ * x + intercept_;
}

void Intersections::Add(const Point &point) {
    auto s1 = std::min(point.seg_main, point.seg_other);
    auto s2 = std::max(point.seg_main, point.seg_other);
    // check if intersection between these two segments is already reported
    auto iter = map_segment_pairs_to_point_.find(std::make_pair(s1, s2));
    // if not, add to map
    if(iter == map_segment_pairs_to_point_.end())
        map_segment_pairs_to_point_.insert(std::make_pair(std::make_pair(s1, s2), point.coordinates()));
}

void Intersections::Print(void) {
    int i = 0;
    for(auto & iter : map_segment_pairs_to_point_) {
        auto pair_intersecting_segments = iter.first;
        auto intersection_coords = iter.second;
        int s1 = pair_intersecting_segments.first;
        int s2 = pair_intersecting_segments.second;

        std::cout << i << ": segs " << s1 << " " << s2 << " intersect at point " << intersection_coords.x() << " "
        << intersection_coords.y()  << std::endl;
        ++i;
    }
}

std::unordered_map<int, std::vector<std::pair<int, Eigen::Vector2d>>> Intersections::getMapSegmentToSegment() {
    std::unordered_map<int, std::vector<std::pair<int, Eigen::Vector2d>>> map_segment_to_segment;
    for (auto &iter : map_segment_pairs_to_point_) {
        auto pair_intersecting_segments = iter.first;
        int s1 = pair_intersecting_segments.first;
        int s2 = pair_intersecting_segments.second;
        // add pair of segment id and intersection point to map
        map_segment_to_segment[s1].push_back(std::make_pair(s2, iter.second));
        map_segment_to_segment[s2].push_back(std::make_pair(s1, iter.second));
    }
    return map_segment_to_segment;
}

bool SweepLineEntry::operator<(const SweepLineEntry &entry) const {
    return y_coord < entry.y_coord;
}

void SweepLine::Print(void) {
    for(auto & entry : vec_entries_)
        std::cout << entry.y_coord << " " << entry.seg_id << std::endl;
}

int SweepLine::FindEntryBySegID(const int seg_id) const {
    for(int i = vec_entries_.size() - 1; i >= 0; --i)
        if(vec_entries_[i].seg_id == seg_id)
            return i;
    return -1;
}

void SweepLine::RemoveEntryAtPos(const int pos) {
    if(pos < 0 || pos >= vec_entries_.size())
        return;
    vec_entries_.erase(vec_entries_.begin() + pos);
}

int SweepLine::GetSuccessorAtPos(const int pos) const {
    if(pos  >= (vec_entries_.size() - 1))
        return -1;

    return vec_entries_[pos + 1].seg_id;
}

int SweepLine::GetPredecessorAtPos(const int pos) const {
    if(pos <= 0)
        return  -1;
    return vec_entries_[pos - 1].seg_id;
}

void SweepLine::InsertEntry(const SweepLineEntry &entry, const double x) {
    //We insert in the right position
    //start from the end and swap until in right position
    //before swapping, update the previous entry value using the sweepline position x

    vec_entries_.push_back(entry);
    int i = vec_entries_.size()  - 1;
    while(i > 0)
    {
        vec_entries_[i-1].y_coord = (*vec_segments)[vec_entries_[i-1].seg_id].ComputeYatX(x);
        if(vec_entries_[i-1].y_coord < vec_entries_[i].y_coord)
            return;

        auto tmp = vec_entries_[i-1];
        vec_entries_[i-1] = vec_entries_[i];
        vec_entries_[i] = tmp;
        --i;
    }
}

void SweepLine::UpdateAndSortEntries(const double x) {
    // update y coordinates at given x position
    for(int i = vec_entries_.size() - 1; i >= 0; --i)
        vec_entries_[i].y_coord = (*vec_segments)[vec_entries_[i].seg_id].ComputeYatX(x);
    // sort entries according to updated y coords
    std::sort(vec_entries_.begin(), vec_entries_.end());
}


static bool CheckIntersection(const SegmentLine &seg_a, const SegmentLine &seg_b, Eigen::Vector2d& intersection_coords) {
    bool intersect = geometry::util::intersectionSegmentSegment(seg_a.pt_left.coordinates(),
                                                                seg_a.pt_right.coordinates(),
                                                                seg_b.pt_left.coordinates(),
                                                                seg_b.pt_right.coordinates(),
                                                                intersection_coords);
    return intersect;
}


void AllPairsIntersections(const std::vector<SegmentLine> & segs, Intersections & intersections) {
    Eigen::Vector2d intersection_coords;
    auto n = segs.size();
    int id = 0;
    for(int i = 0; i < n; ++i)
        for(int j = 0;  j < i; ++j)
            if(CheckIntersection(segs[i], segs[j], intersection_coords))
            {
                Point inter = Point(intersection_coords);
                inter.CreateIntersectionPoint(i, j, id++);
                intersections.Add(inter);
            }
}


void SweepLineIntersections(const std::vector<SegmentLine> & segs,  Intersections  &intersections) {
    // event queue as priority queue
    std::priority_queue< Point > queue;
    std::set<int> removed;
    SweepLine sl;
    SweepLineEntry entry;
    int pred;
    int succ;
    int id = 0;

    std::map<std::pair<int, int>, Point> segsIntersInQueue;
    sl.vec_segments = &segs;

    // initialize event queue
    // points are sorted according to ascending x coordinates (vertical sweep line)
    for(int i = 0; i < (int) segs.size(); ++i)
    {
        // add both points of segment to the event queue
        queue.push(segs[i].pt_left);
        queue.push(segs[i].pt_right);
    }

    // iterate over all entries in the priority queue until queue is empty
    while(!queue.empty()) {
        // get next element from queue as the current point
        auto pt_current = queue.top();
        // delete it from priority queue
        queue.pop();

        // Check if current point is in removed set: If yes, skip point and continue next iteration
        if(removed.find(pt_current.id) != removed.end()) {
            continue;
        }

        /* Main part of algorithm: Handling of events starts here
         * A point should be reported if it is in the intersection of two segments
         * Handling of events is differentiated according to the status of the chosen point
         */

         // Case 1: Point is identified intersection point
        if(pt_current.status == STATUS_INTERSECTION) {
            // Add point to reported intersections
            intersections.Add(pt_current);

            // update and sort entries in SweepLine
            sl.UpdateAndSortEntries(pt_current.x() + Constants::EPSILON);

            auto epos1 = sl.FindEntryBySegID(pt_current.seg_main);
            auto epos2 = sl.FindEntryBySegID(pt_current.seg_other);
            int segLow;
            int segHigh;

            // Order switches at intersection point
            if(epos1 < epos2) {
                segLow = pt_current.seg_other;
                segHigh = pt_current.seg_main;
                succ = sl.GetSuccessorAtPos(epos2);
                pred = sl.GetPredecessorAtPos(epos1);
                //check epos1 with pred and epos2 with succ
            }
            else {
                segLow = pt_current.seg_main;
                segHigh = pt_current.seg_other;
                succ = sl.GetSuccessorAtPos(epos1);
                pred = sl.GetPredecessorAtPos(epos2);
                //check epos1 with succ and epos2 with pred
            }

            if(succ >= 0) {
                Eigen::Vector2d intersection_coords;
                bool intersect = CheckIntersection(segs[segLow], segs[succ], intersection_coords);

                if(intersect && intersection_coords.x() > pt_current.x()) {
                    Point inter(intersection_coords);
                    inter.CreateIntersectionPoint(segLow, succ, id++);
                    queue.push(inter);
                    segsIntersInQueue.insert(
                            std::make_pair(std::make_pair(std::min(inter.seg_main, inter.seg_other),
                                                                           std::max(inter.seg_main, inter.seg_other)),
                                                            inter));
                }
            }

            if(pred >= 0) {
                Eigen::Vector2d intersection_coords;
                bool intersect = CheckIntersection(segs[segHigh], segs[pred], intersection_coords);

                if (intersect && intersection_coords.x() > pt_current.x()) {
                    Point inter(intersection_coords);
                    inter.CreateIntersectionPoint(segHigh, pred, id++);
                    queue.push(inter);
                    segsIntersInQueue.insert(
                            std::make_pair(std::make_pair(std::min(inter.seg_main, inter.seg_other),
                                                          std::max(inter.seg_main, inter.seg_other)),
                                           inter));
                }
            }

            //remove from queue unnecessary events (point C in the reference paper)
            auto iter = segsIntersInQueue.find(std::make_pair(std::min(segHigh, succ), std::max(segHigh, succ)));
            if(succ >= 0 && iter != segsIntersInQueue.end() && iter->second.x() > pt_current.x())
                removed.insert(iter->second.id);
            iter = segsIntersInQueue.find(std::make_pair(std::min(segLow, pred), std::max(segLow, pred)));
            if(pred >= 0 && iter != segsIntersInQueue.end() && iter->second.x() > pt_current.x())
                removed.insert(iter->second.id);
        }
        // Case 2: Point is left point of line segment
        else if(pt_current.status == STATUS_LEFT) {
            //A.1
            // add point to sweep line entries
            entry.y_coord = pt_current.y();
            entry.seg_id = pt_current.seg_main;
            sl.InsertEntry(entry, pt_current.x());

            //A.2
            // find neighboring segments (predecessor and successor) of current segment
            auto epos = sl.FindEntryBySegID(entry.seg_id);
            pred = sl.GetPredecessorAtPos(epos);
            succ =sl.GetSuccessorAtPos(epos);

            if(pred >= 0) {
                // check for intersection with predecessor
                Eigen::Vector2d intersection_coords;
                bool intersect = CheckIntersection(segs[pt_current.seg_main], segs[pred],
                                                   intersection_coords);

                if(intersect && intersection_coords.x() > pt_current.x()) {
                    // if they intersect: add intersection point to priority queue
                    Point inter(intersection_coords);
                    inter.CreateIntersectionPoint(pt_current.seg_main, pred, id++);
                    queue.push(inter);
                    segsIntersInQueue.insert(
                            std::make_pair(std::make_pair(std::min(inter.seg_main, inter.seg_other),
                                                          std::max(inter.seg_main, inter.seg_other)),
                                           inter));
                }
            }

            if(succ >= 0) {
                // check for intersection with successor
                Eigen::Vector2d intersection_coords;
                bool intersect = CheckIntersection(segs[pt_current.seg_main], segs[succ],
                                                   intersection_coords);

                if(intersect && intersection_coords.x() > pt_current.x()) {
                    // if they intersect: add intersection point to priority queue
                    Point inter(intersection_coords);
                    inter.CreateIntersectionPoint(pt_current.seg_main, succ, id++);
                    queue.push(inter);
                    segsIntersInQueue.insert(
                            std::make_pair(std::make_pair(std::min(inter.seg_main, inter.seg_other),
                                                          std::max(inter.seg_main, inter.seg_other)),
                                           inter));
                }
            }

            //A.3
            //removed from queue unnecessary events
            auto iter = segsIntersInQueue.find(std::make_pair(std::min(pred, succ), std::max(pred, succ)));
            if(pred >= 0 && succ >= 0 && iter != segsIntersInQueue.end() && iter->second.x() > pt_current.x())
                removed.insert(iter->second.id);
        }
        // Case 3: point is right point of line segment
        else if(pt_current.status == STATUS_RIGHT) {
            //B.1 & B.2
            // find neighboring segments (predecessor and successor) of current segment
            auto epos = sl.FindEntryBySegID(pt_current.seg_main);
            pred = sl.GetPredecessorAtPos(epos);
            succ = sl.GetSuccessorAtPos(epos);

            //B.3
            // since entry is removed when hitting right point, we need to check if its predecessor and successor
            // intersect
            if(pred >= 0 &&  succ >= 0) {
                Eigen::Vector2d intersection_coords;
                bool intersect = CheckIntersection(segs[pred], segs[succ], intersection_coords);

                if(intersect && intersection_coords.x() > pt_current.x()) {
                    // if they intersect: add intersection point to priority queue
                    Point inter(intersection_coords);
                    inter.CreateIntersectionPoint(pred, succ, id++);
                    queue.push(inter);
                    segsIntersInQueue.insert(
                            std::make_pair(std::make_pair(std::min(inter.seg_main, inter.seg_other),
                                                          std::max(inter.seg_main, inter.seg_other)),
                                           inter));
                }
            }
            // remove segment from active entries in sweep line after hitting the right point
            sl.RemoveEntryAtPos(epos);
        }
    }
}

// TODO remove these methods
// utility functions to create random set of segments for testing purposes
void RandomSeed(const unsigned int s)
{
    srandom(s);
}

unsigned int RandomSeed(void)
{
    FILE  *fp = fopen("/dev/urandom", "r");
    unsigned int s;
    int size = 0;

    if(fp != NULL)
    {
        size = fread(&s, sizeof(unsigned int), 1, fp);
        fclose(fp);
    }
    else
        s = (unsigned int) time(NULL);

    RandomSeed(s);
    return s;
}

/** TODO remove these methods
    RandomUniformReal
    RandomUniformReal
    RandomUniformInteger
    RandomUniformBoolean
 */
static inline double RandomUniformReal(void)
{
    return ((double) random()) / ((double) RANDOM_MAX);
}

static inline double RandomUniformReal(const double min, const double max)
{
    return min + (max - min) * RandomUniformReal();
}

static inline long RandomUniformInteger(const long min, const long max)
{
    const long x = min + (long) RandomUniformReal(0, max - min + 1);
    return x > max ? max : x;
}

static inline bool RandomUniformBoolean(void)
{
    return RandomUniformReal() > 0.5;
}

double DistancePointSegment2D(const double p[2], const double s0[2], const double s1[2])
{
    double pmin[2];
    double a, b;
    const double vx = s1[0] - s0[0];
    const double vy = s1[1] - s0[1];

    if ((a = (vx * (p[0] - s0[0]) + vy * (p[1] - s0[1]))) <= 0)
    {
        pmin[0] = s0[0];
        pmin[1] = s0[1];
    }
    else if ((b = (vx * vx + vy * vy)) <= a)
    {
        pmin[0] = s1[0];
        pmin[1] = s1[1];
    }
    else
    {
        a /= b;
        pmin[0] = s0[0] + a * vx;
        pmin[1] = s0[1] + a * vy;
    }

    return sqrt((p[0] - pmin[0]) * (p[0] - pmin[0]) + (p[1] - pmin[1]) * (p[1] - pmin[1]));
}

// TODO: method not used -> remove
bool AcceptableSegments(const SegmentLine &a, const SegmentLine &b)
{
    double apts[4] =
            {
                    a.pt_left.x(), a.pt_left.y(), a.pt_right.x(), a.pt_right.y()
            };
    double bpts[4] =
            {
                    b.pt_left.x(), b.pt_left.y(), b.pt_right.x(), b.pt_right.y()
            };

    if(DistancePointSegment2D(&apts[0],&bpts[0], &bpts[2]) < Constants::EPSILON ||
       DistancePointSegment2D(&apts[2],&bpts[0], &bpts[2]) < Constants::EPSILON ||
       DistancePointSegment2D(&bpts[0],&apts[0], &apts[2]) < Constants::EPSILON ||
       DistancePointSegment2D(&bpts[0],&apts[0], &apts[2]) < Constants::EPSILON)
        return false;
    return true;
}

/*
 * Functions for the improvement of the curvature
 * The functions below are currently not used within the Sweep Line intersection detection (TODO: Move to separate module?)
 * Archived functions from GR project of Evald
 */
double DistanceSquaredPointSegment2D(const double p[2], const double s0[2], const double s1[2])
{
    double a, b;
    const double vx = s1[0] - s0[0];
    const double vy = s1[1] - s0[1];
    double pmin[2];

    if ((a = (vx * (p[0] - s0[0]) + vy * (p[1] - s0[1]))) <= 0)
    {
        pmin[0] = s0[0];
        pmin[1] = s0[1];
    }
    else if ((b = (vx * vx + vy * vy)) <= a)
    {
        pmin[0] = s1[0];
        pmin[1] = s1[1];
    }
    else
    {
        a /= b;
        pmin[0] = s0[0] + a * vx;
        pmin[1] = s0[1] + a * vy;
    }

    return (p[0] - pmin[0]) * (p[0] - pmin[0]) + (p[1] - pmin[1]) * (p[1] - pmin[1]);
}

bool HelpReduce(const double pts[], const int start, const int end, const double dtol, double &loss)
{
    auto pStart = &pts[2 * start];
    auto pEnd = &pts[2 * end];

    for(int i = start + 1; i < end; ++i)
    {
        auto p = &pts[2 * i];
        double distSquared = DistanceSquaredPointSegment2D(p, pStart, pEnd);
        if(distSquared > dtol * dtol)
            return false;
        loss = loss + distSquared;
    }
    return true;
}

int Reduce(const int n, double pts[], const double dtol, double &loss)
{
    int usen = n;

    for(int start = 0; start < usen; ++start)
        for(int end = usen - 1; end > start; --end)
            // TODO (Idea) Perform a road network intersection check for this "assumed" / "supporting" line.
            // If the line is road network intersection-free (inside road network) then continue removing the points
            // Else, continue to next iteration

            // Optimize a function ---> (sum of dSqauredDistance) and (number of points)
            if(HelpReduce(pts, start, end, dtol, loss))
            {
                int k = 1;
                for(int i = end; i < usen; ++i) //delete intermediate points from start + 1 to end - 1.
                {
                    pts[2 * (start + k)] = pts[2 * i];
                    pts[2 * (start + k) + 1] = pts[2 * i + 1];
                    ++k;
                }
                usen = usen - (end - start - 1);
                std::cout << "yes delete start = " << start << " end = " << end << " usen = " << usen << std::endl;
                break;

            }
            else
                std::cout << "no delete start = " << start << " end = " << end << " usen = " << usen << std::endl;

    return usen;
}

}  // namespace sweep_line_util
}  // namespace geometry



