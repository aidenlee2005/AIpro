#include <iostream>
#include <sstream>
#include <string>

extern std::vector<std::string> pythonoutput;

#ifdef COLOR_STDOUT
#define BOLD   "\033[1m"
#define RED    "\033[31m"
#define GREEN  "\033[32m"
#define YELLOW "\033[33m"
#define BLUE   "\033[34m"
#define RESET  "\033[0m"
#else
#define BOLD   ""
#define RED    ""
#define GREEN  ""
#define YELLOW ""
#define BLUE   ""
#define RESET  ""
#endif

#ifdef PYTEST_ON
#define PYTEST  //nothing happens
#else
#define PYTEST  return
#endif

std::vector<std::string> run_python_split(const std::string& cmd, char delim = '#') {

    std::cout << BOLD << RED << "Running python..." << RESET << std::endl;
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return {};
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }
    pclose(pipe);

    // 分割字符串
    std::vector<std::string> parts;
    std::stringstream ss(result);
    std::string item;
    while (std::getline(ss, item, delim)) {
        // 可选：去除首尾空格
        size_t start = item.find_first_not_of(" \n\r\t");
        size_t end = item.find_last_not_of(" \n\r\t");
        if (start != std::string::npos && end != std::string::npos)
            item = item.substr(start, end - start + 1);
        else
            item = "";
        if (!item.empty())
            parts.push_back(item);
    }
    return parts;
}

void runPy(int idx){
    PYTEST;
    std::cout << BOLD << GREEN << "**Pytorch Result**" << RESET << std::endl;
    std::cout << GREEN << pythonoutput[idx] << RESET << std::endl;
    std::cout << BOLD << GREEN << "**CORRECT**" << RESET << std::endl;
    std::cout << std::endl;
}