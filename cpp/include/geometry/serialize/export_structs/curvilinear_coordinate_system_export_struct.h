#pragma once
namespace geometry {
namespace serialize {
struct CurvilinearCoordinateSystemExportStruct {
 public:
  double default_projection_domain_limit;
  double eps;
  double eps2;
  int method;
};
}  // namespace serialize
}  // namespace geometry
