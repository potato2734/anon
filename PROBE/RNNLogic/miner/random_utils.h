#ifndef RANDOM_UTILS_H
#define RANDOM_UTILS_H

#include <random>
#include <cstdlib>  // For srand() and rand()

// Global random engine
inline std::mt19937& getGlobalRNG() {
    static std::mt19937 rng;
    return rng;
}

// Function to set a custom seed
inline void setGlobalSeed(unsigned int seed) {
    getGlobalRNG().seed(seed);   // Set seed for mt19937
    srand(seed);                 // Set seed for rand()
}

#endif // RANDOM_UTILS_H
