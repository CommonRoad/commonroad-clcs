#ifndef CLCS_LOGGER_H
#define CLCS_LOGGER_H

#include <unordered_map>

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"


namespace geometry {

    class CLCSLogger {
    public:
        /**
         * Static utility class for handling logger of the Curvilinear Coordinate System
         */

        // Delete constructor, copy constructor, assignment and destructor
        CLCSLogger() = delete;
        CLCSLogger(const CLCSLogger &) = delete;
        CLCSLogger &operator=(const CLCSLogger &) = delete;
        ~CLCSLogger() = delete;

        /**
         * Init: register logger with spdlog
         */
        static void init(const std::string &str_log_level = "off") {
            // get spdlog level
            spdlog::level::level_enum log_level = getLogLevelfromStr_(str_log_level);

            // if logger does not yet exist
            if ( !spdlog::get(LOGGER_NAME) ) {
                auto logger = spdlog::stdout_color_mt(LOGGER_NAME);
                logger->set_pattern("[%^%l%$] [%n] %v");
                logger->set_level(log_level);
            }
            // if logger exists, update log level
            else {
                auto logger = spdlog::get(LOGGER_NAME);
                logger->set_level(log_level);
            }
        }

        /**
         * Getter for logger
         */
        static std::shared_ptr<spdlog::logger> getLogger() {
            // ensure CLCS logger exists
            if ( !spdlog::get(LOGGER_NAME) ) {
                init();
            }
            return spdlog::get(LOGGER_NAME);
        }

    private:
        static constexpr const char* LOGGER_NAME = "CLCS_Logger";

        static inline const std::unordered_map<std::string , spdlog::level::level_enum> map_str_to_log_levels_ = {
            {"trace", spdlog::level::trace},
            {"debug", spdlog::level::debug},
            {"info", spdlog::level::info},
            {"warn", spdlog::level::warn},
            {"err", spdlog::level::err},
            {"critical", spdlog::level::critical},
            {"off", spdlog::level::off}
        };

        /**
         * Returns spdlog level from log level given as string.
         * Defaults to spdlog::level::off if the provided level does not match.
         * @param log_level given as string
         * @return log level as type spdlog::level::level_enum
         */
        static spdlog::level::level_enum getLogLevelfromStr_(const std::string &log_level) {
            auto it = map_str_to_log_levels_.find(log_level);
            if (it != map_str_to_log_levels_.end()) {
                return it->second;
            } else {
                return spdlog::level::off;
            }
        }

    };

} // namespace geometry

#endif //CLCS_LOGGER_H
