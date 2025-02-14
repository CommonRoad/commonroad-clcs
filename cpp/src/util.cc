#include "geometry/util.h"
namespace geometry {
namespace util {

Eigen::VectorXd computeCurvature(const geometry::EigenPolyline &polyline) {
    // convert to RowMatrix
    RowMatrixXd polyline_rows;
    geometry::util::to_RowMatrixXd(polyline, polyline_rows);

    // get x and y coords
    Eigen::VectorXd x_coords{polyline_rows.middleCols(0, 1).eval()};
    Eigen::VectorXd y_coords{polyline_rows.middleCols(1, 1).eval()};

    // compute pathlength vector
    Eigen::VectorXd pathlength = computePathlength(polyline_rows);

    // compute gradients via central differences
    Eigen::VectorXd x_d = gradient(x_coords, pathlength);
    Eigen::VectorXd x_dd = gradient(x_d, pathlength);
    Eigen::VectorXd y_d = gradient(y_coords, pathlength);
    Eigen::VectorXd y_dd = gradient(y_d, pathlength);
    Eigen::VectorXd curvature =
            ((x_d.cwiseProduct(y_dd).eval() - x_dd.cwiseProduct(y_d).eval()).array() /
             ((x_d.array().pow(2) + y_d.array().pow(2)).pow(3. / 2.)))
                    .eval();

    return curvature;
}

Eigen::VectorXd computeCurvature(const geometry::EigenPolyline &polyline, int digits) {
    auto curvature = computeCurvature(polyline);
    int vec_size = curvature.size();
    double precision = pow(10.0, digits);

    Eigen::VectorXd vec_curvature_rounded(vec_size);
    for (int i = 0; i < vec_size; i++) {
        // round value to precision
        double curvature_rounded = std::round(curvature[i] * precision) / precision;
        if (std::fabs(curvature_rounded) < pow(10.0, -digits)) {
            vec_curvature_rounded[i] = std::fabs(curvature_rounded);
        } else {
            vec_curvature_rounded[i] = curvature_rounded;
        }
        vec_curvature_rounded[i] = curvature_rounded;
    }
    return vec_curvature_rounded;
}

Eigen::VectorXd computePathlength(const geometry::EigenPolyline &polyline) {
    // convert to RowMatrix
    RowMatrixXd polyline_rows;
    geometry::util::to_RowMatrixXd(polyline, polyline_rows);

    return computePathlength(polyline_rows);
}

Eigen::VectorXd computePathlength(const RowMatrixXd &polyline_rows) {
    // get x and y coords
    Eigen::VectorXd x_coords{polyline_rows.middleCols(0, 1).eval()};
    Eigen::VectorXd y_coords{polyline_rows.middleCols(1, 1).eval()};

    // compute pathl ength vector
    Eigen::VectorXd pathlength(x_coords.size());
    pathlength[0] = 0.0; // First point doesn't have a previous one to compare
    for (int i = 1; i < x_coords.size(); ++i) {
        double dx = x_coords[i] - x_coords[i - 1];
        double dy = y_coords[i] - y_coords[i - 1];
        pathlength[i] = pathlength[i - 1] + std::sqrt(dx * dx + dy * dy);
    }
    return pathlength;
}

std::vector<int> getInflectionPointsIdx(const EigenPolyline &polyline, int digits) {
    // TODO Improve algorithm:
    // TODO precision: when digits are too low, no inflection points are detected
    // TODO resulting partitions still might have changes in the sign of the curvature
    auto vec_curvature = computeCurvature(polyline, digits);
    std::vector<int> idx_inflection_points;

    for (int i = 1; i < vec_curvature.size(); i++) {
        if (std::fpclassify(vec_curvature[i - 1]) != FP_ZERO && std::fpclassify(vec_curvature[i]) != FP_ZERO && vec_curvature[i - 1] * vec_curvature[i] < 0) {
            idx_inflection_points.push_back(i - 1);
        }
    }
    return idx_inflection_points;
}

void computePathPartitions(const EigenPolyline &path_input,
                           std::vector<EigenPolyline> &path_partitions) {
    // get inflection points
    int digits = 4;
    std::vector<int> idx_inflection_pts = util::getInflectionPointsIdx(path_input, digits);

    if (idx_inflection_pts.empty()) {
        path_partitions.push_back(path_input);
    } else {
        // first partition
        EigenPolyline first_partition(path_input.begin(),
                                      path_input.begin() + idx_inflection_pts.front() + 1);
        path_partitions.push_back(first_partition);

        // in between partitions
        for (int i = 0; i < idx_inflection_pts.size() - 1; i++) {
            EigenPolyline partition(path_input.begin() + idx_inflection_pts[i],
                                    path_input.begin() + idx_inflection_pts[i + 1] + 1);
            path_partitions.push_back(partition);
        }

        //last partition
        EigenPolyline last_partition(path_input.begin() + idx_inflection_pts.back(),
                                     path_input.end());
        path_partitions.push_back(last_partition);
    }
}

int resample_polyline(const RowMatrixXd& polyline, double step,
                      geometry::EigenPolyline& new_polyline) {
  if (polyline.rows() < 2) {
    to_EigenPolyline(polyline, new_polyline);
    return 0;
  }

  new_polyline = geometry::EigenPolyline();
  new_polyline.push_back(polyline.row(0).transpose().eval());
  double current_position = step;
  double current_length = (polyline.row(0) - polyline.row(1)).norm();
  int current_idx = 0;
  while (current_idx < (polyline.rows() - 1)) {
    if (current_position >= current_length) {
      current_position = current_position - current_length;
      current_idx += 1;
      if (current_idx > polyline.rows() - 2) break;
      current_length = (polyline.row(current_idx + 1) - polyline.row(current_idx)).norm();
    } else {
      double rel = current_position / current_length;
      new_polyline.push_back(
          ((1 - rel) * polyline.row(current_idx) + rel * polyline.row(current_idx + 1))
              .eval()
              .transpose()
              .eval());
      current_position += step;
    }
  }

  Eigen::Vector2d last = polyline.bottomRows<1>().transpose();

  // avoid duplicating last point due to precision errors
  if ((last - new_polyline.back()).norm() >= 1e-6) {
      new_polyline.push_back(last.eval());
  }

  return 0;
}

int resample_polyline(const geometry::EigenPolyline& polyline, double step,
                      geometry::EigenPolyline& ret)

{
  RowMatrixXd polyline_converted;
  to_RowMatrixXd(polyline, polyline_converted);
  return resample_polyline(polyline_converted, step, ret);
}

int resample_polyline(const RowMatrixXd &polyline, double step,
                      RowMatrixXd &ret) {
  geometry::EigenPolyline new_polyline;
  int err = resample_polyline(polyline, step, new_polyline);
  to_RowMatrixXd(new_polyline, ret);
  return err;
}

void chaikins_corner_cutting(const RowMatrixXd &polyline, int refinements,
                            RowMatrixXd &ret) {
  RowMatrixXd el2 = polyline.eval();
  for (int cc1 = 0; cc1 < refinements; cc1++) {
    RowMatrixXd el(2 * el2.rows(), el2.cols());

    int j = 0;

    for (int i = 0; i < el2.rows(); ++i) {
      const int k = 2;
      el.middleRows(j, 2) = el2.row(i).colwise().replicate(2);
      j += k;
    }

    RowMatrixXd R = RowMatrixXd::Zero(el.rows(), el.cols());

    R.row(0) = el.row(0);

    for (int cc1 = 2; cc1 < R.rows(); cc1 += 2) {
      R.row(cc1) = el.row(cc1 - 1);
      R.row(cc1 - 1) = el.row(cc1);
    }
    R.row(R.rows() - 1) = el.row(el.rows() - 1);

    el2 = (el * 0.75 + R * 0.25).eval();
  }

  ret = el2;
}

void chaikins_corner_cutting(const geometry::EigenPolyline &polyline,
                            int refinements, geometry::EigenPolyline &ret) {
  // convert to RowMatrixXd type
  RowMatrixXd polyline_matrix;
  to_RowMatrixXd(polyline, polyline_matrix);
  // init return matrix
  RowMatrixXd ret_matrix;
  // subdivision
  chaikins_corner_cutting(polyline_matrix, refinements, ret_matrix);
  // convert to return polyline
  to_EigenPolyline(ret_matrix, ret);
}

void lane_riesenfeld_subdivision(const RowMatrixXd &polyline_matrix,
                                 int degree,
                                 int refinements,
                                 RowMatrixXd &ret_matrix) {
    // set initial return matrix
    ret_matrix = polyline_matrix;
    // init temp matrices
    RowMatrixXd subdivided_matrix;

    for (int cnt = 0; cnt < refinements; cnt++) {
        // Chaikin's subdivision step: vertex duplication + one averaging step
        chaikins_corner_cutting(ret_matrix, 1, subdivided_matrix);

        // init counter: set to 1, because the Chaikin's step already contains one averaging step
        int i{1};
        while (i < degree) {
            // one averaging step
            auto num_rows{subdivided_matrix.rows()};
            // compute average between two consecutive points
            RowMatrixXd averaged_pts = (subdivided_matrix.topRows(num_rows - 1) +
                    subdivided_matrix.bottomRows(num_rows - 1)) / 2.0;

            // resize matrix by one row
            subdivided_matrix.conservativeResize(num_rows + 1, subdivided_matrix.cols());

            // fill updated subdivided matrix
            // first point of original polyline
            subdivided_matrix.topRows(1) = ret_matrix.topRows(1);
            // fill averaged points
            subdivided_matrix.middleRows(1, averaged_pts.rows() + 1) = averaged_pts.eval();
            // last point of original polyline
            subdivided_matrix.bottomRows(1) = ret_matrix.bottomRows(1);

            i++;
        }
        // set return matrix
        ret_matrix = subdivided_matrix;
    }
}

int to_RowMatrixXd(const geometry::EigenPolyline& polyline, RowMatrixXd& ret) {
  ret = RowMatrixXd(polyline.size(), 2);
  for (int cc1 = 0; cc1 < polyline.size(); cc1++) {
    ret.row(cc1) = polyline[cc1].transpose().eval();
  }
  return 0;
}

int to_EigenPolyline(const RowMatrixXd& polyline,
                     geometry::EigenPolyline& ret) {
  ret = geometry::EigenPolyline();
  for (int cc1 = 0; cc1 < polyline.rows(); cc1++) {
    ret.push_back(polyline.row(cc1).transpose().eval());
  }
  return 0;
}

Eigen::VectorXd gradient(const Eigen::VectorXd &input,
                         const Eigen::VectorXd &coordinates) {
    if (input.size() <= 1) return input;

    Eigen::VectorXd res(input.size());

    // first central differences for first and last point
    res[0] = (input[1] - input[0]) / (coordinates[1] - coordinates[0]);
    int last_idx = res.size() - 1;
    res[last_idx] = (input[last_idx] - input[last_idx - 1]) / (coordinates[last_idx] - coordinates[last_idx - 1]);

    // second central differences for other points
    for (int j = 1; j < input.size() - 1; j++) {
        int j_l = j - 1;
        int j_r = j + 1;
        // compute gradient
        double grad{(input[j_r] - input[j_l]) / (coordinates[j_r] - coordinates[j_l])};
        res[j] = grad;
    }
    return res;
}

}  // namespace util
}  // namespace geometry
