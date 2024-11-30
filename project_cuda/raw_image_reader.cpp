#include <fstream>
#include <iostream>
#include <vector>
#include <cstring>
#include "raw_image_reader.hpp"

bool raw::readImageRAW(const std::string& filename, unsigned char*& buffer, long long& width, long long& height) {
    // Otwórz plik w trybie binarnym
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Nie udało się otworzyć pliku: " << filename << std::endl;
        return false;
    }
    
    file.read(reinterpret_cast<char*>(&width), sizeof(long long));
    file.read(reinterpret_cast<char*>(&height), sizeof(long long));

    // Przykładowe wymiary obrazu (musisz znać je z góry dla RAW)
    long long size = width * height;

    // Alokuj pamięć
    buffer = new unsigned char[size];

    // Czytaj dane do bufora
    file.read(reinterpret_cast<char*>(buffer), size);

    // Sprawdź, czy wszystkie dane zostały odczytane
    if (!file) {
        std::cerr << "Błąd podczas czytania pliku lub plik jest za mały." << std::endl;
        delete[] buffer;
        buffer = nullptr;
        return false;
    }

    file.close();
    return true;
}

void raw::writeImageRAW(const std::string& filename, unsigned char* buffer, long long width, long long height) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Nie udało się zapisać pliku: " << filename << std::endl;
        return;
    }

    file.write(reinterpret_cast<const char*>(&width), sizeof(long long));
    file.write(reinterpret_cast<const char*>(&height), sizeof(long long));

    long long size = width * height;
    file.write(reinterpret_cast<const char*>(buffer), size);

    file.close();
}