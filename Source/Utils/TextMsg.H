/* Copyright 2022 Luca Fedeli
 *
 * This file is part of Artemis.
 *
 * License: BSD-3-Clause-LBNL
 */

#ifndef ARTEMIS_TEXT_MSG_H_
#define ARTEMIS_TEXT_MSG_H_

#include <ablastr/utils/TextMsg.H>

#define ARTEMIS_ALWAYS_ASSERT_WITH_MESSAGE(EX,MSG) (EX)?((void)0) : ablastr::utils::TextMsg::Assert( # EX , __FILE__, __LINE__ , MSG)

#define ARTEMIS_ABORT_WITH_MESSAGE(MSG) ablastr::utils::TextMsg::Abort( __FILE__, __LINE__ , MSG)

namespace Utils::TextMsg
{
    using namespace ablastr::utils::TextMsg;
} // namespace Utils::TextMsg

#endif // ARTEMIS_TEXT_MSG_H_
