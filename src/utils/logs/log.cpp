
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>

#include "log4cpp/Category.hh"
#include "log4cpp/Appender.hh"
#include "log4cpp/FileAppender.hh"
#include "log4cpp/OstreamAppender.hh"
#include "log4cpp/Layout.hh"
#include "log4cpp/BasicLayout.hh"
#include "log4cpp/Priority.hh"

using namespace log4cpp;

Category& log_console = Category::getRoot();
Category& log_file = Category::getInstance(std::string("log_file"));

namespace log4cpp {

	Priority::PriorityLevel priority = Priority::DEBUG;
	//Priority::PriorityLevel priority = Priority::INFO;

	void initLogs() {

		log_console.setPriority(priority);
		log_file.setPriority(priority);

		log4cpp::Appender *appender_console = new log4cpp::OstreamAppender("console", &std::cout);
		appender_console->setLayout(new log4cpp::BasicLayout());
		appender_console->setThreshold(priority);
		log_console.addAppender(appender_console);

		log4cpp::Appender *appender_file = new log4cpp::FileAppender("default", "program.log");
		appender_file->setLayout(new log4cpp::BasicLayout());
		appender_file->setThreshold(priority);
		//log_file.addAppender(appender_file);

		//console redirected to file
		log_console.addAppender(appender_file);

	}
}
