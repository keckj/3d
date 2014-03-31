#pragma once

#include "log4cpp/Category.hh"


extern log4cpp::Category& log_console;
extern log4cpp::Category& log_file;

namespace log4cpp {
	void initLogs();
}
