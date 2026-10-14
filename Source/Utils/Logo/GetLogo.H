#ifndef ARTEMIS_GET_LOGO_H_
#define ARTEMIS_GET_LOGO_H_

#include <string>

namespace utils
{
    namespace logo
    {
        /**
         * \brief provides an ASCII art logo for Artemis
         *
         * \return a string containing an ASCII art logo
         */
        std::string get_logo () noexcept;
    }
}

#endif
