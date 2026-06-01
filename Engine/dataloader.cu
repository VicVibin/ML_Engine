#include "dataloader.h"

#ifdef _WIN32
Text FolderPaths(const str& folder, const int filenums) 
{
    Text files;

    str search_path = folder + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE hFind = FindFirstFileA(search_path.c_str(), &fd);

    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                files.emplace_back(fd.cFileName);
            }
        } while (FindNextFileA(hFind, &fd));
        FindClose(hFind);
    }
    if(filenums <= 0) return files;

    if (static_cast<int>(files.size()) < filenums)
    {
        std::cout << "Not enough files in directory... num files: " << files.size() << "\n";
        if(files.size() > 0 ) for(const auto &i : files) std::cout << i << "\t";
    }

    if(static_cast<int>(files.size()) == filenums)
    {
        return files;
    }

    Text new_files;
    for(int i =0; i < filenums;++i) new_files.push_back(files[i]);

    return new_files;
}

#else
Text FolderPaths(const str& folder, const int filenums)
{
    Text files;
 
    DIR* dir = opendir(folder.c_str());
    if (!dir) {
        std::cerr << "Cannot open directory: " << folder << "\n";
        return files;
    }
 
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        // Skip the "." and ".." pseudo-entries
        if (entry->d_name[0] == '.') continue;
 
        str fullPath = folder + "/" + entry->d_name;
 
        struct stat st{};
        if (stat(fullPath.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            files.emplace_back(entry->d_name);
        }
    }
    closedir(dir);
    if (filenums <= 0) return files;
 
    if (static_cast<int>(files.size()) < filenums) {
        std::cout << "Not enough files in directory... num files: "
                  << files.size() << "\n";
        for (const auto& f : files) std::cout << f << "\t";
    }
 
    if (static_cast<int>(files.size()) == filenums) return files;
 
    // Trim or return whatever we have
    Text new_files;
    for (int i = 0; i < filenums && i < static_cast<int>(files.size()); ++i)
        new_files.push_back(files[i]);
 
    return new_files;
}
#endif
