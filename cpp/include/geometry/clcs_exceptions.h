#ifndef CLCS_EXCEPTIONS_H
#define CLCS_EXCEPTIONS_H

#include <stdexcept>

namespace geometry {


/**
 * Error for wrong computation method value for projection domain computation
 */
class InvalidMethodError : public std::invalid_argument {
public:
    explicit InvalidMethodError() : std::invalid_argument("Invalid method for projection domain computation:"
                                                          "Valid inputs: 1, 2") {}
};


/**
 * Error for invalid reference path to construct CLCS
 */
class InvalidReferencePathError : public std::invalid_argument {
public:
    explicit InvalidReferencePathError() : std::invalid_argument("Invalid reference path for CLCS construction:"
                                                                 "Reference path must have at least 3 points") {}
};


/**
 * Projection domain errors are thrown when trying to convert coordinates
 * outside the projection domain.
 */
class ProjectionDomainError : public std::invalid_argument {
public:
    explicit ProjectionDomainError(const std::string& message)
            : std::invalid_argument(message) {}
};


/**
 * General exception for curvilinear coordinate outside the projection domain, either laterally or longitudinally.
 */
class CurvilinearProjectionDomainError : public ProjectionDomainError {
public:
    explicit CurvilinearProjectionDomainError()
            : ProjectionDomainError("Longitudinal and/or lateral coordinate outside of curvilinear "
                                    "projection domain") {}
};


/**
 * Specific exception for longitudinal coordinate outside the curvilinear projection domain.
 */
class CurvilinearProjectionDomainLongitudinalError : public ProjectionDomainError {
public:
    explicit CurvilinearProjectionDomainLongitudinalError()
            : ProjectionDomainError("Longitudinal coordinate outside of curvilinear projection domain") {}
};


/**
 * Specific exception for lateral coordinate outside the projection domain.
 */
class CurvilinearProjectionDomainLateralError : public ProjectionDomainError {
public:
    explicit CurvilinearProjectionDomainLateralError()
            : ProjectionDomainError("Lateral coordinate outside of curvilinear projection domain") {}
};


/**
 * General exception for cartesian coordinates outside the Cartesian projection domain.
 */
class CartesianProjectionDomainError : public ProjectionDomainError {
public:
    explicit CartesianProjectionDomainError()
            : ProjectionDomainError("x and/or y coordinate outside of Cartesian projection domain") {}
};



} // namespace geometry

#endif //CLCS_EXCEPTIONS_H
