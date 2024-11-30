#include <vector>
#include <cstring>

namespace raw {
    bool readImageRAW(const std::string& filename, unsigned char*& buffer, long long& width, long long& height);

    void writeImageRAW(const std::string& filename, unsigned char* buffer, long long width, long long height);  
}
