#include "RCL.hpp"
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <array>
#include <chrono>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <unordered_set>
#include <cmath>
#include <numeric>
#include "unembeddable_graphs.h"

#define DEBUG_PRINT(x) if (print_statement) { std::cout << "DEBUG: " << x << std::endl; }

#define WORDSIZE 64
#define MAXNV 64
#define DEFAULT_PARTITION_SIZE 0

// Default number of unembeddable graphs to check (overridden by RCL.hpp)
// #define DEFAULT_UNEMBEDDABLE_CHECK 13

// Add these two variables for optimization
//const int SMALL_GRAPH_ORDER = 13;
//const int EDGE_CUTOFF = 78;  // 13 choose 2 = 78

// Add these constants for the predefined vectors at the top of the file with other constants
//const int NUM_PREDEFINED_VECTORS = 13;

// Remove these constants as they'll be set dynamically
// const int SMALL_GRAPH_ORDER = 25;
// const int EDGE_CUTOFF = 300;
// const int NUM_PREDEFINED_VECTORS = 25;

const int VECTOR_DIMENSION = 3;

long canon = 0;
long noncanon = 0;
double canontime = 0;
double noncanontime = 0;
long canonarr[MAXNV] = {};
long noncanonarr[MAXNV] = {};
double canontimearr[MAXNV] = {};
double noncanontimearr[MAXNV] = {};
long muscount = 0;
long muscounts[17] = {0};  // Initialize all elements to zero explicitly
double mustime = 0;
// Add at the top with other file pointers
// FILE* solution_file = fopen("solutions.txt", "w");

// Add at the top with other time variables
double subgraph_check_time = 0;
double orthogonality_check_time = 0;
long orthogonality_violations = 0;

// Add this to the top with other counters
std::unordered_set<size_t> seen_orthogonality_assignments;
long orthogonality_checks = 0;
long orthogonality_skipped = 0;

// Add this to the top with other boolean flags
bool use_orthogonality = false;

SymmetryBreaker::SymmetryBreaker(CaDiCaL::Solver* s, int order, int uc, int ps, const std::string& vectors_file) : solver(s) {
    // Set print_statement to false by default (can be enabled with --print-statement)
    print_statement = false;
    
    DEBUG_PRINT("Entered RCL SymmetryBreaker constructor");
    if (order == 0) {
        std::cout << "c Need to provide order to use programmatic code" << std::endl;
        return;
    }
    
    // Set default unembeddable check if not specified
    if (uc < 0) {
        // Not specified on command line, use default
        uc = DEFAULT_UNEMBEDDABLE_CHECK;
        std::cout << "c Using default unembeddable subgraphs check (" << uc << " graphs)" << std::endl;
    } else if (uc == 0) {
        // If explicitly set to 0, disable the check
        std::cout << "c Disabling unembeddable subgraphs check" << std::endl;
    } else {
        std::cout << "c Checking for " << uc << " unembeddable subgraphs" << std::endl;
    }
    
    // Store the unembeddable check value
    unembeddable_check = uc;
    
    // Set default partition size if not specified
    if (ps == 0) {
        ps = DEFAULT_PARTITION_SIZE;
        std::cout << "c Using default partition size (" << ps << ")" << std::endl;
    } else {
        std::cout << "c Using partition size " << ps << std::endl;
    }
    
    n = order;
    num_edge_vars = n*(n-1)/2;
    partition_size = ps;
    
    // Set SMALL_GRAPH_ORDER and NUM_PREDEFINED_VECTORS to partition_size
    SMALL_GRAPH_ORDER = partition_size;
    NUM_PREDEFINED_VECTORS = partition_size;
    
    // Set EDGE_CUTOFF to partition_size choose 2
    EDGE_CUTOFF = partition_size * (partition_size - 1) / 2;
    
    // Add a clear print statement showing both order and partition size
    std::cout << "c Running with order = " << n << " and partition size = " << partition_size << std::endl;
    std::cout << "c SMALL_GRAPH_ORDER = " << SMALL_GRAPH_ORDER << ", NUM_PREDEFINED_VECTORS = " << NUM_PREDEFINED_VECTORS << std::endl;
    std::cout << "c EDGE_CUTOFF = " << EDGE_CUTOFF << " (partition_size choose 2)" << std::endl;
    
    DEBUG_PRINT("Allocating memory");
    assign = new int[num_edge_vars];
    fixed = new bool[num_edge_vars];
    colsuntouched = new int[n];
    
    DEBUG_PRINT("Connecting external propagator");
    solver->connect_external_propagator(this);
    
    DEBUG_PRINT("Initializing arrays");
    for (int i = 0; i < num_edge_vars; i++) {
        assign[i] = l_Undef;
        fixed[i] = false;
        solver->add_observed_var(i+1);
    }
    std::fill_n(colsuntouched, n, 0);
    
    current_trail.push_back(std::vector<int>());
    
    learned_clauses_count = 0;
    canonize_calls = 0;
    total_canonize_time = 0;
    
    seen_partial_assignments.clear();
    
    initNautyOptions();

    // Do not load the master graph by default
    //load_master_graph("cadical-rcl/data/SI-C-c1-labeled-37.lad");
    //load_master_graph("cadical-rcl/data/SI-C-c2-labeled-853-13.lad");
    //load_master_graph("cadical-rcl/data/SI-C-c2-labeled-853-25.lad");

    // Reset counters
    muscount = 0;
    for (int i = 0; i < 17; i++) {
        muscounts[i] = 0;
    }

    // Initialize use_orthogonality to false by default
    use_orthogonality = false;

    // Initialize predefined vectors
    if (!vectors_file.empty()) {
        // Load vectors from file
        load_predefined_vectors_from_file(vectors_file);
    } else {
        // Use default vectors
        // OLD 13-vertex vectors - COMMENTED OUT
        /*
        predefined_vectors = {
            {-1, 1, 1},   // v1
            {1, -1, 1},   // v2
            {1, 1, -1},   // v3
            {1, 1, 1},    // v4
            {1, 0, 0},    // v5
            {0, 1, 0},    // v6
            {0, 0, 1},    // v7
            {1, -1, 0},   // v8
            {1, 0, -1},   // v9
            {0, 1, -1},   // v10
            {0, 1, 1},    // v11
            {1, 0, 1},    // v12
            {1, 1, 0}     // v13
        };
        */
        
        // NEW 25-vertex vectors based on canonical ordering (real default)
        predefined_vectors = {
            {-2, 1, 1},   // v1  -> Position 1  -> Vertex 16
            {1, -2, 1},   // v2  -> Position 2  -> Vertex 23
            {-1, 2, 1},   // v3  -> Position 3  -> Vertex 20
            {2, -1, 1},   // v4  -> Position 4  -> Vertex 15
            {-1, 1, 2},   // v5  -> Position 5  -> Vertex 25
            {2, 1, -1},   // v6  -> Position 6  -> Vertex 17
            {1, 2, 1},    // v7  -> Position 7  -> Vertex 24
            {1, -1, 2},   // v8  -> Position 8  -> Vertex 22
            {1, 2, -1},   // v9  -> Position 9  -> Vertex 18
            {2, 1, 1},    // v10 -> Position 10 -> Vertex 14
            {1, 1, -2},   // v11 -> Position 11 -> Vertex 21
            {1, 1, 2},    // v12 -> Position 12 -> Vertex 19
            {1, 0, 0},    // v13 -> Position 13 -> Vertex 1
            {0, 1, 0},    // v14 -> Position 14 -> Vertex 2
            {1, -1, 0},   // v15 -> Position 15 -> Vertex 9
            {-1, 1, 1},   // v16 -> Position 16 -> Vertex 10
            {1, -1, 1},   // v17 -> Position 17 -> Vertex 11
            {0, 0, 1},    // v18 -> Position 18 -> Vertex 3
            {1, 1, -1},   // v19 -> Position 19 -> Vertex 12
            {1, 0, -1},   // v20 -> Position 20 -> Vertex 7
            {0, 1, -1},   // v21 -> Position 21 -> Vertex 4
            {0, 1, 1},    // v22 -> Position 22 -> Vertex 5
            {1, 1, 1},    // v23 -> Position 23 -> Vertex 13
            {1, 0, 1},    // v24 -> Position 24 -> Vertex 6
            {1, 1, 0}     // v25 -> Position 25 -> Vertex 8
        };
        
        // Empty complex defaults; can be set via set_predefined_vectors_complex_from_strings
        predefined_vectors_complex.clear();
        predefined_vectors_cq.clear();
        
        // Ensure we have enough predefined vectors
        while (predefined_vectors.size() < NUM_PREDEFINED_VECTORS) {
            // Add more vectors if needed
            predefined_vectors.push_back({1, 1, 1});
        }
    }

    // Initialize permutation file
    perm_file = nullptr;
    perm_filename = "";

    // Initialize orthogonality file
    ortho_file = nullptr;
    ortho_filename = "";

    DEBUG_PRINT("SymmetryBreaker constructor completed");
}

SymmetryBreaker::~SymmetryBreaker() {
    // First disconnect the propagator
    solver->disconnect_external_propagator();
    
    // Free memory
    delete[] assign;
    delete[] fixed;
    delete[] colsuntouched;
    
    // Remove the solution file closing code
    // if (solution_file) {
    //     fclose(solution_file);
    //     solution_file = nullptr;
    // }
    
    // Force flush stdout before printing statistics
    fflush(stdout);
    
    // Print all statistics with explicit fflush after each section
    printf("Number of solutions   : %ld\n", sol_count);
    fflush(stdout);
    
    printf("Canonical subgraphs   : %-12ld   (%.0f /sec)\n", canon, canon > 0 ? canon/canontime : 0);
    fflush(stdout);
    
    for(int i=2; i<n; i++) {
        printf("          order %2d    : %-12ld   (%.0f /sec)\n", 
               i+1, 
               canonarr[i], 
               canonarr[i] > 0 ? canonarr[i]/canontimearr[i] : 0);
        fflush(stdout);
    }
    
    printf("Noncanonical subgraphs: %-12ld   (%.0f /sec)\n", noncanon, noncanon > 0 ? noncanon/noncanontime : 0);
    fflush(stdout);
    
    for(int i=2; i<n; i++) {
        printf("          order %2d    : %-12ld   (%.0f /sec)\n", 
               i+1, 
               noncanonarr[i], 
               noncanonarr[i] > 0 ? noncanonarr[i]/noncanontimearr[i] : 0);
        fflush(stdout);
    }
    
    printf("Canonicity checking   : %g s\n", canontime);
    printf("Noncanonicity checking: %g s\n", noncanontime);
    printf("Total canonicity time : %g s\n", canontime + noncanontime);
    fflush(stdout);
    
    if (unembeddable_check > 0) {
        printf("Unembeddable checking : %g s\n", mustime);
        fflush(stdout);
        
        for(int g=0; g<unembeddable_check; g++) {
            printf("        graph #%2d     : %-12ld\n", g, muscounts[g]);
            fflush(stdout);
        }
        
        printf("Total unembed. graphs : %ld\n", muscount);
        fflush(stdout);
    }

    // Print subgraph statistics
    print_subgraph_statistics();
    fflush(stdout);
    
    // Final flush to ensure everything is printed
    fflush(stdout);
    
    // Close permutation file if open
    if (perm_file) {
        fclose(perm_file);
        perm_file = nullptr;
    }

    // Close orthogonality file if open
    if (ortho_file) {
        fclose(ortho_file);
        ortho_file = nullptr;
    }
}

std::string SymmetryBreaker::convert_assignment_to_string(int k) {
    const int size = k * (k - 1) / 2;
    std::string result;
    result.reserve(size);  // Pre-allocate exact size needed
    
    // Use more efficient character access
    for (int j = 1; j < k; j++) {
        for (int i = 0; i < j; i++) {
            result.push_back((assign[j*(j-1)/2 + i] == l_True) ? '1' : '0');
        }
    }
    return result;
}

void SymmetryBreaker::stringToGraph(const std::string& input, graph* g, int n, int m) {
    int index = 0;
    DEBUG_PRINT("Converting string to graph. Input: " << input);
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++) {
            if (input[index++] == '1') {
                ADDONEEDGE(g, i, j, m);
                ADDONEEDGE(g, j, i, m);  // Ensure symmetry
                DEBUG_PRINT("Added edge between " << i << " and " << j);
            }
        }
    }
}

void SymmetryBreaker::Getcan_Rec(graph g[MAXNV], int n, int can[], int orbits[]) {
    int lab1[MAXNV], lab2[MAXNV], inv_lab1[MAXNV], ptn[MAXNV];
    int i, j, k;
    setword st;
    graph g2[MAXNV];
    int m = SETWORDSNEEDED(n);

    if (n == 1) {
        can[n-1] = n-1;
    } else {
        // Set up nauty options
        options.writeautoms = FALSE;
        options.writemarkers = FALSE;
        options.getcanon = TRUE;
        
        // Update to use the partition_size member variable
        if (n > partition_size) {
            options.defaultptn = FALSE;
            for (i = 0; i < n-1; i++) {
                if (i == partition_size-1) {
                    ptn[i] = 0;  // Mark end of first partition
                } else {
                    ptn[i] = 1;  // Same partition continues
                }
            }
            ptn[n-1] = 0;  // Last vertex always ends its partition
        } else {
            options.defaultptn = TRUE;  // Use default partition for small graphs
        }

        // Initialize lab array with identity permutation
        for (i = 0; i < n; i++) {
            lab1[i] = i;
        }

        nauty(g, lab1, ptn, NULL, orbits, &options, &stats, workspace, 50, m, n, g2);

        for (i = 0; i < n; i++)
            inv_lab1[lab1[i]] = i;
        for (i = 0; i <= n-2; i++) {
            j = lab1[i];
            st = g[j];
            g2[i] = 0;
            while (st) {
                k = FIRSTBIT(st);
                st ^= bit[k];
                k = inv_lab1[k];
                if (k != n-1)
                    g2[i] |= bit[k];
            }
        }
        Getcan_Rec(g2, n-1, lab2, orbits);
        for (i = 0; i <= n-2; i++)
            can[i] = lab1[lab2[i]];
        can[n-1] = lab1[n-1];
    }
}

bool SymmetryBreaker::isCanonical(const std::string& input) {
    int n = static_cast<int>(std::sqrt(2 * input.length() + 0.25) + 0.5);
    int m = SETWORDSNEEDED(n);
    
    DEBUG_PRINT("Checking canonicity for input: " << input);
    DEBUG_PRINT("Calculated n = " << n << ", m = " << m);

    graph g[MAXNV];
    for (int i = 0; i < n; i++) {
        EMPTYSET(GRAPHROW(g, i, m), m);
    }
    stringToGraph(input, g, n, m);

    DEBUG_PRINT("Original graph:");
    printGraph(g, n, m);

    int can[MAXNV];
    graph cang[MAXNV];
    
    Getcan_Rec(g, n, can, orbits);

    if (print_statement) {
        std::cout << "DEBUG: Canonical labeling:" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << "  " << can[i] << " -> " << i << std::endl;
        }
    }

    // Construct the canonical graph
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cang[i*m + j] = 0;
        }
        for (int j = 0; j < n; j++) {
            if (ISELEMENT(GRAPHROW(g, can[i], m), can[j])) {
                ADDONEEDGE(cang, i, j, m);
            }
        }
    }

    DEBUG_PRINT("Canonical graph:");
    printGraph(cang, n, m);

    // Compare the input graph with the canonical graph
    bool is_canonical = true;
    for (int i = 0; i < n && is_canonical; i++) {
        for (int j = 0; j < m && is_canonical; j++) {
            if (g[i*m + j] != cang[i*m + j]) {
                is_canonical = false;
            }
        }
    }

    DEBUG_PRINT("isCanonical result: " << (is_canonical ? "true" : "false"));
    return is_canonical;
}

std::pair<bool, std::vector<int>> SymmetryBreaker::isCanonicalWithPermutation(const std::string& input) {
    int n = static_cast<int>(std::sqrt(2 * input.length() + 0.25) + 0.5);
    int m = SETWORDSNEEDED(n);
    
    DEBUG_PRINT("Checking canonicity for input: " << input);
    DEBUG_PRINT("Calculated n = " << n << ", m = " << m);

    graph g[MAXNV];
    for (int i = 0; i < n; i++) {
        EMPTYSET(GRAPHROW(g, i, m), m);
    }
    stringToGraph(input, g, n, m);

    DEBUG_PRINT("Original graph:");
    printGraph(g, n, m);

    int can[MAXNV];
    graph cang[MAXNV];
    
    Getcan_Rec(g, n, can, orbits);

    if (print_statement) {
        std::cout << "DEBUG: Canonical labeling:" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << "  " << can[i] << " -> " << i << std::endl;
        }
    }

    // Construct the canonical graph
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            cang[i*m + j] = 0;
        }
        for (int j = 0; j < n; j++) {
            if (ISELEMENT(GRAPHROW(g, can[i], m), can[j])) {
                ADDONEEDGE(cang, i, j, m);
            }
        }
    }

    DEBUG_PRINT("Canonical graph:");
    printGraph(cang, n, m);

    // Compare the input graph with the canonical graph
    bool is_canonical = true;
    for (int i = 0; i < n && is_canonical; i++) {
        for (int j = 0; j < m && is_canonical; j++) {
            if (g[i*m + j] != cang[i*m + j]) {
                is_canonical = false;
            }
        }
    }

    // Convert can[] array to permutation vector
    // can[i] gives the original vertex that maps to canonical position i
    // We want permutation[i] to give the canonical position that original vertex i maps to
    std::vector<int> permutation(n);
    for (int i = 0; i < n; i++) {
        permutation[can[i]] = i;
    }

    DEBUG_PRINT("isCanonical result: " << (is_canonical ? "true" : "false"));
    if (!is_canonical && print_statement) {
        DEBUG_PRINT("Permutation (original -> canonical): ");
        for (int i = 0; i < n; i++) {
            std::cout << permutation[i];
        }
        std::cout << std::endl;
    }

    return std::make_pair(is_canonical, permutation);
}

bool SymmetryBreaker::has_mus_subgraph(int k, int* P, int* p, int g) {
    int pl[12]; // pl[k] contains the current list of possibilities for kth vertex
    int pn[13]; // pn[k] contains the initial list of possibilities for kth vertex
    pl[0] = (1 << k) - 1;
    pn[0] = (1 << k) - 1;
    int i = 0;

    while(1) {
        // If no possibilities for ith vertex then backtrack
        if(pl[i]==0) {
            while((pl[i] & (pl[i] - 1)) == 0) {
                i--;
                if(i==-1) {
                    return false;  // No permutations produce a matrix containing the gth submatrix
                }
            }
            pl[i] = pl[i] & ~(1 << p[i]);
        }

        p[i] = log2(pl[i] & -pl[i]); // Get index of rightmost high bit
        pn[i+1] = pn[i] & ~(1 << p[i]); // List of possibilities for (i+1)th vertex

        // Check if permuted matrix contains the gth submatrix
        bool result_known = false;
        for(int j=0; j<i; j++) {
            if(!mus[g][i*(i-1)/2+j]) continue;
            const int px = MAX(p[i], p[j]);
            const int py = MIN(p[i], p[j]);
            const int pj = px*(px-1)/2 + py;
            if(assign[pj] == l_False) {
                result_known = true;
                break;
            }
        }

        if(!result_known && ((i == 9 && g < 2) || (i == 10 && g < 7) || i == 11)) {
            // Found complete gth submatrix in p(M)
            for(int j=0; j<=i; j++) {
                P[p[j]] = j;
            }
            return true;
        }
        if(!result_known) {
            i++;
            pl[i] = pn[i];
        } else {
            pl[i] = pl[i] & ~(1 << p[i]);
        }
    }
}

std::vector<int> SymmetryBreaker::call_RCL_binary(const std::string& input, int k) {
        auto result = isCanonicalWithPermutation(input);
        bool is_canonical = result.first;
        std::vector<int> permutation = result.second;
        
        if (!is_canonical) {
            if (print_statement) {
                std::cout << "DEBUG: Input is not canonical" << std::endl;
            }
            
            // Write permutation to file if filename is set
            if (!perm_filename.empty()) {
                write_permutation_to_file(permutation);
            }
            
            std::vector<int> blocking_clause = generate_naive_blocking_clause(input);
            
            // Add blocking clause as trusted clause to DRAT proof (like cadical-ks)
            solver->add_trusted_clause(blocking_clause);
            
            return blocking_clause;
        }
    
    // Check for unembeddable subgraphs
    if (unembeddable_check > 0) {
        int P[n];
        int p[12];
        for(int j=0; j<n; j++) P[j] = -1;
        
        const double before = CaDiCaL::absolute_process_time();
        for(int g=0; g<unembeddable_check; g++) {
            if (has_mus_subgraph(k, P, p, g)) {
                muscount++;
                muscounts[g]++;
                
                // Generate blocking clause for unembeddable subgraph
                std::vector<int> blocking_clause;
                int c = 0;
                for(int jj=0; jj<k; jj++) {
                    for(int ii=0; ii<jj; ii++) {
                        if(input[c] == '1' && P[jj] != -1 && P[ii] != -1) {
                            if((P[ii] < P[jj] && mus[g][P[ii] + P[jj]*(P[jj]-1)/2]) || 
                               (P[jj] < P[ii] && mus[g][P[jj] + P[ii]*(P[ii]-1)/2])) {
                                blocking_clause.push_back(-(c+1));
                            }
                        }
                        c++;
                    }
                }
                const double after = CaDiCaL::absolute_process_time();
                mustime += (after-before);
                
                if (print_statement) {
                    std::cout << "DEBUG: Found unembeddable subgraph #" << g << std::endl;
                }
                return blocking_clause;
            }
        }
    }
    
    if (print_statement) {
        std::cout << "DEBUG: Input is canonical" << std::endl;
    }
    return std::vector<int>();
}

// Helper function to extract a submatrix from the input string
std::string SymmetryBreaker::extract_submatrix(const std::string& input, int k) {
    std::string submatrix;
    int n = static_cast<int>(std::sqrt(2 * input.length() + 0.25) + 0.5);
    for (int j = 1; j < k; ++j) {
        for (int i = 0; i < j; ++i) {
            int index = j * (j - 1) / 2 + i;
            submatrix += input[index];
        }
    }
    return submatrix;
}

// Add this new function to check for non-decreasing degrees
std::vector<int> SymmetryBreaker::check_non_decreasing_degrees(const std::string& assignment, int k) {
    std::vector<int> blocking_clause;
    
    // Only proceed if we have at least 2 columns
    if (k < 2) return blocking_clause;
    
    // Count ones in the last two columns only
    int second_last_ones = 0;
    int last_ones = 0;
    
    // Count ones in second-last column (k-2)
    for (int i = 0; i < k-2; i++) {
        int var_index = (k-2)*(k-3)/2 + i;
        if (assignment[var_index] == '1') {
            second_last_ones++;
        }
    }
    
    // Count ones in last column (k-1)
    for (int i = 0; i < k-1; i++) {
        int var_index = (k-1)*(k-2)/2 + i;
        if (assignment[var_index] == '1') {
            last_ones++;
        }
    }

    if (print_statement) {
        std::cout << "DEBUG: Last two column degrees: " 
                  << second_last_ones << " " << last_ones << std::endl;
    }

    // Check if second-last column has more ones than last column
    if (second_last_ones > last_ones) {
        if (print_statement) {
            std::cout << "DEBUG: Columns " << k-2 << " and " << k-1 
                     << " violate non-decreasing order (" 
                     << second_last_ones << " > " << last_ones << ")" << std::endl;
        }
        
        // Add literals for second-last column
        for (int i = 0; i < k-2; i++) {
            int var_index = (k-2)*(k-3)/2 + i;
            int literal = assignment[var_index] == '1' ? -(var_index + 1) : (var_index + 1);
            blocking_clause.push_back(literal);
        }
        
        // Add literals for last column
        for (int i = 0; i < k-1; i++) {
            int var_index = (k-1)*(k-2)/2 + i;
            int literal = assignment[var_index] == '1' ? -(var_index + 1) : (var_index + 1);
            blocking_clause.push_back(literal);
        }

        if (print_statement) {
            std::cout << "DEBUG: Generated degree blocking clause: ";
            for (int lit : blocking_clause) {
                std::cout << lit << " ";
            }
            std::cout << std::endl;
        }
    }

    return blocking_clause;
}

// Add this function to compute the dot product of two vectors
int SymmetryBreaker::dot_product(const std::vector<int>& v1, const std::vector<int>& v2) {
    int result = 0;
    for (size_t i = 0; i < v1.size(); i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

// Add this function to compute the cross product of two 3D vectors
std::vector<int> SymmetryBreaker::cross_product(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> result(3);
    result[0] = a[1] * b[2] - a[2] * b[1];
    result[1] = a[2] * b[0] - a[0] * b[2];
    result[2] = a[0] * b[1] - a[1] * b[0];
    return result;
}

// Placeholder complex dot product; to be implemented later
int SymmetryBreaker::dot_complex(const std::vector<int>& v1, const std::vector<int>& v2) {
    // Supports two encodings:
    // 1) Real-only: length == VECTOR_DIMENSION (imag parts assumed 0)
    // 2) Interleaved complex: length == 2*VECTOR_DIMENSION as [Re0, Im0, Re1, Im1, Re2, Im2]
    // Returns 0 iff Hermitian inner product (conj(v1)·v2) equals 0 as a complex number.

    const bool v1_is_complex = v1.size() == static_cast<size_t>(2 * VECTOR_DIMENSION);
    const bool v2_is_complex = v2.size() == static_cast<size_t>(2 * VECTOR_DIMENSION);

    long long real_sum = 0;
    long long imag_sum = 0;

    for (int idx = 0; idx < VECTOR_DIMENSION; idx++) {
        // v1 component (a + i b), but conjugated: (a - i b)
        long long a = v1_is_complex ? v1[2 * idx] : v1[idx];
        long long b = v1_is_complex ? v1[2 * idx + 1] : 0;

        // v2 component (c + i d)
        long long c = v2_is_complex ? v2[2 * idx] : v2[idx];
        long long d = v2_is_complex ? v2[2 * idx + 1] : 0;

        // (a - i b)*(c + i d) = (ac + bd) + i(ad - bc)
        real_sum += a * c + b * d;
        imag_sum += a * d - b * c;
    }

    // Orthogonal iff both sums are zero
    if (real_sum == 0 && imag_sum == 0) return 0;

    // Return a non-zero sentinel for violation (prefer real part if non-zero)
    return real_sum != 0 ? static_cast<int>(real_sum) : static_cast<int>(imag_sum);
}

// Placeholder complex cross product; to be implemented later
std::vector<int> SymmetryBreaker::cross_complex(const std::vector<int>& a, const std::vector<int>& b) {
    // Supports two encodings:
    // 1) Real-only: length == VECTOR_DIMENSION
    // 2) Interleaved complex: length == 2*VECTOR_DIMENSION as [Re0, Im0, Re1, Im1, Re2, Im2]
    // Hermitian cross product (x, y, z):
    //   x = conj(a2) * b3 - conj(a3) * b2
    //   y = conj(a3) * b1 - conj(a1) * b3
    //   z = conj(a1) * b2 - conj(a2) * b1

    const bool a_is_complex = a.size() == static_cast<size_t>(2 * VECTOR_DIMENSION);
    const bool b_is_complex = b.size() == static_cast<size_t>(2 * VECTOR_DIMENSION);

    // Real-only fallback equals standard cross product
    if (!a_is_complex && !b_is_complex) {
        return cross_product(a, b);
    }

    auto get_re = [](const std::vector<int>& v, bool complex, int idx) -> long long {
        return complex ? static_cast<long long>(v[2 * idx]) : static_cast<long long>(v[idx]);
    };
    auto get_im = [](const std::vector<int>& v, bool complex, int idx) -> long long {
        return complex ? static_cast<long long>(v[2 * idx + 1]) : 0LL;
    };

    auto mul_conj_a_b = [&](int ai, int bi) -> std::pair<long long, long long> {
        // conj(a_ai) * b_bi
        long long a_re = get_re(a, a_is_complex, ai);
        long long a_im = get_im(a, a_is_complex, ai);
        long long b_re = get_re(b, b_is_complex, bi);
        long long b_im = get_im(b, b_is_complex, bi);
        // (a_re - i a_im) * (b_re + i b_im) = (a_re*b_re + a_im*b_im) + i(a_re*b_im - a_im*b_re)
        long long re = a_re * b_re + a_im * b_im;
        long long im = a_re * b_im - a_im * b_re;
        return {re, im};
    };

    // x = conj(a2)*b3 - conj(a3)*b2
    auto x1 = mul_conj_a_b(1, 2);
    auto x2 = mul_conj_a_b(2, 1);
    long long xr = x1.first - x2.first;
    long long xi = x1.second - x2.second;

    // y = conj(a3)*b1 - conj(a1)*b3
    auto y1 = mul_conj_a_b(2, 0);
    auto y2 = mul_conj_a_b(0, 2);
    long long yr = y1.first - y2.first;
    long long yi = y1.second - y2.second;

    // z = conj(a1)*b2 - conj(a2)*b1
    auto z1 = mul_conj_a_b(0, 1);
    auto z2 = mul_conj_a_b(1, 0);
    long long zr = z1.first - z2.first;
    long long zi = z1.second - z2.second;

    std::vector<int> result(2 * VECTOR_DIMENSION);
    result[0] = static_cast<int>(xr);
    result[1] = static_cast<int>(xi);
    result[2] = static_cast<int>(yr);
    result[3] = static_cast<int>(yi);
    result[4] = static_cast<int>(zr);
    result[5] = static_cast<int>(zi);
    return result;
}

// Parse a single complex token like "1", "-1", "I", "-I", "-I - 1", "I + 1"
// Returns pair {real, imag} as ints
std::pair<int,int> SymmetryBreaker::parse_complex_component(const std::string& token_in) {
    // Normalize by removing spaces
    std::string t;
    t.reserve(token_in.size());
    for (char c : token_in) if (c != ' ') t.push_back(c);

    // Handle pure I forms
    if (t == "I") return {0, 1};
    if (t == "+I") return {0, 1};
    if (t == "-I") return {0, -1};

    // If contains 'I'
    size_t posI = t.find('I');
    if (posI != std::string::npos) {
        // Split around '+' or '-' separating real and imag
        // Try patterns: a+I, a-Il, I+a, -I+a, I-a, -I-a, a+I*b (only unit coeffs allowed here)
        int real = 0, imag = 0;

        // Find the last '+' or '-' excluding leading sign
        size_t split = std::string::npos;
        for (size_t i = 1; i < t.size(); i++) {
            if (t[i] == '+' || t[i] == '-') split = i;
        }

        if (split == std::string::npos) {
            // Imag only like I or -I already handled, or malformed
            // Fallback: 0 + 1*I
            imag = (t[0] == '-') ? -1 : 1;
            return {real, imag};
        }

        std::string left = t.substr(0, split);
        std::string right = t.substr(split); // includes sign

        // Determine which side contains 'I'
        if (left.find('I') != std::string::npos) {
            // left imag, right real
            // left is "+I" or "-I"
            imag = (left[0] == '-') ? -1 : 1;
            real = std::stoi(right);
        } else {
            // left real, right imag (like +I or -I)
            real = std::stoi(left);
            imag = (right[0] == '-') ? -1 : 1;
        }
        return {real, imag};
    }

    // Pure real
    return {std::stoi(t), 0};
}

// Set complex predefined vectors from string triples, converting to interleaved ints
void SymmetryBreaker::set_predefined_vectors_complex_from_strings(const std::vector<std::array<std::string,3>>& spec) {
    predefined_vectors_complex.clear();
    predefined_vectors_complex.reserve(spec.size());
    for (const auto& triple : spec) {
        auto c0 = parse_complex_component(triple[0]);
        auto c1 = parse_complex_component(triple[1]);
        auto c2 = parse_complex_component(triple[2]);
        std::vector<int> v(2 * VECTOR_DIMENSION);
        v[0] = c0.first; v[1] = c0.second;
        v[2] = c1.first; v[3] = c1.second;
        v[4] = c2.first; v[5] = c2.second;
        predefined_vectors_complex.push_back(std::move(v));
    }
}

// Rational helpers
static int64_t gcd(int64_t a, int64_t b) {
    while (b != 0) {
        int64_t temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

SymmetryBreaker::Rational SymmetryBreaker::rational_normalize(Rational r) const {
    if (r.den < 0) { r.den = -r.den; r.num = -r.num; }
    if (r.num == 0) { r.den = 1; return r; }
    int64_t g = gcd(std::llabs(r.num), std::llabs(r.den));
    if (g > 1) { r.num /= g; r.den /= g; }
    return r;
}

bool SymmetryBreaker::rational_is_zero(const Rational &x) const { return x.num == 0; }

SymmetryBreaker::Rational SymmetryBreaker::rational_add(const Rational &x, const Rational &y) const {
    Rational r{ x.num * y.den + y.num * x.den, x.den * y.den };
    return rational_normalize(r);
}
SymmetryBreaker::Rational SymmetryBreaker::rational_sub(const Rational &x, const Rational &y) const {
    Rational r{ x.num * y.den - y.num * x.den, x.den * y.den };
    return rational_normalize(r);
}
SymmetryBreaker::Rational SymmetryBreaker::rational_mul(const Rational &x, const Rational &y) const {
    Rational r{ x.num * y.num, x.den * y.den };
    return rational_normalize(r);
}
SymmetryBreaker::Rational SymmetryBreaker::rational_neg(const Rational &x) const {
    Rational r{ -x.num, x.den };
    return r;
}

// Quadratic helpers over Q(√rad): x = a + b√rad
bool SymmetryBreaker::quadratic_is_zero(const Quadratic &q) const {
    return rational_is_zero(q.a) && rational_is_zero(q.b);
}

SymmetryBreaker::Quadratic SymmetryBreaker::quadratic_add(const Quadratic &x, const Quadratic &y) const {
    Quadratic r{ rational_add(x.a, y.a), rational_add(x.b, y.b), x.rad };
    return r;
}
SymmetryBreaker::Quadratic SymmetryBreaker::quadratic_sub(const Quadratic &x, const Quadratic &y) const {
    Quadratic r{ rational_sub(x.a, y.a), rational_sub(x.b, y.b), x.rad };
    return r;
}
SymmetryBreaker::Quadratic SymmetryBreaker::quadratic_mul(const Quadratic &x, const Quadratic &y) const {
    // (a1 + b1√r)(a2 + b2√r) = (a1a2 + b1b2 r) + (a1b2 + a2b1)√r
    Rational a1a2 = rational_mul(x.a, y.a);
    Rational b1b2 = rational_mul(x.b, y.b);
    Rational rterm{ b1b2.num * y.rad, b1b2.den };
    Rational a = rational_add(a1a2, rterm);
    Rational b = rational_add(rational_mul(x.a, y.b), rational_mul(y.a, x.b));
    Quadratic out{ rational_normalize(a), rational_normalize(b), x.rad };
    return out;
}

SymmetryBreaker::Quadratic SymmetryBreaker::quadratic_conj_rad(const Quadratic &x) const {
    // Conjugation over i does not change √rad; this is identity for our use (only Complex conj flips im)
    return x;
}

// ComplexQuadratic helpers
SymmetryBreaker::ComplexQuadratic SymmetryBreaker::cq_conj(const ComplexQuadratic &z) const {
    ComplexQuadratic out = z;
    out.im.a = rational_neg(out.im.a);
    out.im.b = rational_neg(out.im.b);
    return out;
}

SymmetryBreaker::ComplexQuadratic SymmetryBreaker::cq_add(const ComplexQuadratic &x, const ComplexQuadratic &y) const {
    return ComplexQuadratic{ quadratic_add(x.re, y.re), quadratic_add(x.im, y.im) };
}
SymmetryBreaker::ComplexQuadratic SymmetryBreaker::cq_sub(const ComplexQuadratic &x, const ComplexQuadratic &y) const {
    return ComplexQuadratic{ quadratic_sub(x.re, y.re), quadratic_sub(x.im, y.im) };
}
SymmetryBreaker::ComplexQuadratic SymmetryBreaker::cq_mul(const ComplexQuadratic &x, const ComplexQuadratic &y) const {
    // (xr + i xi)(yr + i yi) = (xr*yr - xi*yi) + i(xr*yi + xi*yr)
    Quadratic real = quadratic_sub(quadratic_mul(x.re, y.re), quadratic_mul(x.im, y.im));
    Quadratic imag = quadratic_add(quadratic_mul(x.re, y.im), quadratic_mul(x.im, y.re));
    return ComplexQuadratic{ real, imag };
}
bool SymmetryBreaker::cq_is_zero(const ComplexQuadratic &z) const {
    return quadratic_is_zero(z.re) && quadratic_is_zero(z.im);
}

std::array<SymmetryBreaker::ComplexQuadratic,3> SymmetryBreaker::cross_complex_cq(
    const std::array<ComplexQuadratic,3>& a,
    const std::array<ComplexQuadratic,3>& b) const {
    // x = conj(a2)*b3 - conj(a3)*b2
    ComplexQuadratic x = cq_sub(cq_mul(cq_conj(a[1]), b[2]), cq_mul(cq_conj(a[2]), b[1]));
    // y = conj(a3)*b1 - conj(a1)*b3
    ComplexQuadratic y = cq_sub(cq_mul(cq_conj(a[2]), b[0]), cq_mul(cq_conj(a[0]), b[2]));
    // z = conj(a1)*b2 - conj(a2)*b1
    ComplexQuadratic z = cq_sub(cq_mul(cq_conj(a[0]), b[1]), cq_mul(cq_conj(a[1]), b[0]));
    return {x, y, z};
}

SymmetryBreaker::ComplexQuadratic SymmetryBreaker::dot_complex_cq(
    const std::array<ComplexQuadratic,3>& a,
    const std::array<ComplexQuadratic,3>& b) const {
    ComplexQuadratic sum{ Quadratic{ Rational{0,1}, Rational{0,1}, 2 }, Quadratic{ Rational{0,1}, Rational{0,1}, 2 } };
    for (int i = 0; i < 3; i++) {
        sum = cq_add(sum, cq_mul(cq_conj(a[i]), b[i]));
    }
    return sum;
}

// Parsing helpers for exact sqrt
SymmetryBreaker::Rational SymmetryBreaker::rational_from_string(const std::string& s) {
    // format: integer or a/b
    std::string t; t.reserve(s.size());
    for (char c : s) if (c != ' ') t.push_back(c);
    size_t slash = t.find('/');
    if (slash == std::string::npos) return rational_normalize(Rational{ std::stoll(t), 1 });
    std::string ns = t.substr(0, slash);
    std::string ds = t.substr(slash + 1);
    return rational_normalize(Rational{ std::stoll(ns), std::stoll(ds) });
}

SymmetryBreaker::Quadratic SymmetryBreaker::parse_quadratic_real(const std::string& token) {
    // Accept forms: r, r*sqrt(n), +/- combinations with a single sqrt term: a + b*sqrt(n), a - b*sqrt(n)
    std::string t; t.reserve(token.size()); for (char c : token) if (c != ' ') t.push_back(c);
    size_t sq = t.find("sqrt(");
    if (sq == std::string::npos) {
        return Quadratic{ rational_from_string(t), Rational{0,1}, 2 };
    }
    // Split at + or - between terms
    size_t split = std::string::npos;
    for (size_t i = 1; i < t.size(); i++) if (t[i] == '+' || t[i] == '-') split = i;
    Rational a{0,1}; Rational b{0,1}; int rad = 2;
    if (split == std::string::npos) {
        // pure b*sqrt(n)
        std::string coef = t.substr(0, sq);
        if (coef.empty() || coef == "+") b = Rational{1,1};
        else if (coef == "-") b = Rational{-1,1};
        else b = rational_from_string(coef);
    } else {
        std::string left = t.substr(0, split);
        std::string right = t.substr(split);
        if (left.find("sqrt(") != std::string::npos) {
            // left is b*sqrt(n), right is a
            size_t open = left.find('('), close = left.find(')');
            std::string coef = left.substr(0, sq);
            if (coef.empty() || coef == "+") b = Rational{1,1};
            else if (coef == "-") b = Rational{-1,1};
            else b = rational_from_string(coef);
            a = rational_from_string(right);
        } else {
            // left is a, right is ±b*sqrt(n)
            a = rational_from_string(left);
            std::string s2 = right;
            size_t sq2 = s2.find("sqrt(");
            std::string coef = s2.substr(0, sq2);
            if (coef == "+" || coef.empty()) b = Rational{1,1};
            else if (coef == "-") b = Rational{-1,1};
            else b = rational_from_string(coef);
        }
    }
    // Extract radicand n
    size_t open = t.find('('), close = t.find(')');
    if (open != std::string::npos && close != std::string::npos && close > open + 1) {
        rad = std::stoi(t.substr(open + 1, close - open - 1));
    }
    return Quadratic{ rational_normalize(a), rational_normalize(b), rad };
}

SymmetryBreaker::ComplexQuadratic SymmetryBreaker::parse_cq_component(const std::string& token) {
    // Accept combined forms in a single token, e.g. "1+I", "-1/2*sqrt(2)-I", "3/4+1/2*sqrt(3)+(-1/3)*I"
    // Strategy: split token (whitespace removed) into real and imag parts by isolating occurrences of I.
    std::string t; t.reserve(token.size()); for (char c : token) if (c != ' ') t.push_back(c);

    // If no 'I', imag = 0, real = quadratic real parser
    if (t.find('I') == std::string::npos) {
        Quadratic re = parse_quadratic_real(t);
        Quadratic im{ Rational{0,1}, Rational{0,1}, re.rad };
        return ComplexQuadratic{ re, im };
    }

    // Split into pieces separated by '+' or '-' but keep signs; then aggregate terms with and without 'I'
    std::vector<std::string> terms;
    size_t i = 0; size_t n = t.size();
    while (i < n) {
        size_t j = i;
        if (t[j] == '+' || t[j] == '-') j++;
        while (j < n && t[j] != '+' && t[j] != '-') j++;
        terms.push_back(t.substr(i, j - i));
        i = j;
    }

    std::string real_accum;
    std::string imag_accum;
    for (const auto& term : terms) {
        if (term.find('I') != std::string::npos) {
            // Strip trailing 'I'
            std::string coef = term;
            // allow forms like "+I", "-I"
            if (coef == "I" || coef == "+I") coef = "+1";
            else if (coef == "-I") coef = "-1";
            else {
                // remove trailing I
                if (!coef.empty() && coef.back() == 'I') coef.pop_back();
                // if now ends with '*' (e.g., "1/2*sqrt(2)*"), drop
                if (!coef.empty() && coef.back() == '*') coef.pop_back();
            }
            if (!imag_accum.empty() && coef[0] != '+' && coef[0] != '-') imag_accum += "+";
            imag_accum += coef;
        } else {
            if (!real_accum.empty() && term[0] != '+' && term[0] != '-') real_accum += "+";
            real_accum += term;
        }
    }

    Quadratic re = real_accum.empty() ? Quadratic{ Rational{0,1}, Rational{0,1}, 2 } : parse_quadratic_real(real_accum);
    Quadratic im = imag_accum.empty() ? Quadratic{ Rational{0,1}, Rational{0,1}, 2 } : parse_quadratic_real(imag_accum);
    return ComplexQuadratic{ re, im };
}

void SymmetryBreaker::set_predefined_vectors_cq_from_strings(const std::vector<std::array<std::string,3>>& spec) {
    predefined_vectors_cq.clear();
    predefined_vectors_cq.reserve(spec.size());
    for (const auto& triple : spec) {
        ComplexQuadratic c0 = parse_cq_component(triple[0]);
        ComplexQuadratic c1 = parse_cq_component(triple[1]);
        ComplexQuadratic c2 = parse_cq_component(triple[2]);
        std::array<ComplexQuadratic,3> v = {c0, c1, c2};
        predefined_vectors_cq.push_back(std::move(v));
    }
}

void SymmetryBreaker::load_predefined_vectors_from_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open vectors file " << filepath << std::endl;
        std::cerr << "Falling back to default vectors" << std::endl;
        return;
    }

    std::string line;
    std::vector<std::array<std::string, 3>> vector_specs;
    
    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string comp1, comp2, comp3;
        
        if (iss >> comp1 >> comp2 >> comp3) {
            vector_specs.push_back({comp1, comp2, comp3});
        } else {
            std::cerr << "Warning: Skipping malformed line: " << line << std::endl;
        }
    }
    
    if (vector_specs.empty()) {
        std::cerr << "Warning: No valid vectors found in file " << filepath << std::endl;
        std::cerr << "Falling back to default vectors" << std::endl;
        return;
    }
    
    // Determine if these are complex vectors (contain 'I' or 'sqrt')
    bool has_complex = false;
    for (const auto& spec : vector_specs) {
        for (const auto& comp : spec) {
            if (comp.find('I') != std::string::npos || comp.find("sqrt(") != std::string::npos) {
                has_complex = true;
                break;
            }
        }
        if (has_complex) break;
    }
    
    if (has_complex) {
        // Load as complex vectors with exact sqrt support
        set_predefined_vectors_cq_from_strings(vector_specs);
        std::cout << "c Loaded " << vector_specs.size() << " complex vectors from " << filepath << std::endl;
    } else {
        // Load as simple integer vectors
        predefined_vectors.clear();
        predefined_vectors_complex.clear();
        predefined_vectors_cq.clear();
        
        for (const auto& spec : vector_specs) {
            std::vector<int> vec(3);
            vec[0] = std::stoi(spec[0]);
            vec[1] = std::stoi(spec[1]);
            vec[2] = std::stoi(spec[2]);
            predefined_vectors.push_back(vec);
        }
        
        std::cout << "c Loaded " << vector_specs.size() << " integer vectors from " << filepath << std::endl;
    }
    
    // Ensure we have enough predefined vectors
    while (predefined_vectors.size() < NUM_PREDEFINED_VECTORS) {
        predefined_vectors.push_back({1, 1, 1});
    }
}

// Modify the check_orthogonality_constraints function to correctly calculate edge indices
std::vector<int> SymmetryBreaker::check_orthogonality_constraints(const std::string& assignment, int k) {
    const double start_time = CaDiCaL::absolute_process_time();
    std::vector<int> blocking_clause;
    
    // Only proceed if we have at least 14 vertices (13 predefined + at least 1 to check)
    if (k <= NUM_PREDEFINED_VECTORS) {
        orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
        return blocking_clause;
    }
    
    // Check if we've already processed this assignment
    size_t hash_value = std::hash<std::string>{}(assignment);
    if (seen_orthogonality_assignments.find(hash_value) != seen_orthogonality_assignments.end()) {
        if (print_statement) {
            DEBUG_PRINT("Skipping already processed orthogonality assignment of size " << k);
        }
        orthogonality_skipped++;
        orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
        return blocking_clause;
    }
    
    // Add this assignment to the seen set
    seen_orthogonality_assignments.insert(hash_value);
    orthogonality_checks++;
    
    if (print_statement) {
        DEBUG_PRINT("Checking orthogonality constraints for " << k << " vertices");
    }
    
    // Convert assignment string to adjacency matrix
    adjacency_matrix_t matrix = string_to_adjacency_matrix(assignment, k);
    
    // Decide computation mode
    const bool cq_mode = use_complex && static_cast<int>(predefined_vectors_cq.size()) >= NUM_PREDEFINED_VECTORS;
    
    // Storage for integer/complex-int mode
    std::vector<std::vector<int>> vectors(
        k,
        use_complex && !cq_mode ? std::vector<int>(2 * VECTOR_DIMENSION, 0)
                                : std::vector<int>(VECTOR_DIMENSION, 0)
    );
    // Storage for exact sqrt complex mode
    std::vector<std::array<ComplexQuadratic,3>> vectors_cq;
    if (cq_mode) vectors_cq.resize(k);
    
    std::vector<bool> vector_assigned(k, false);
    
    // Track dependencies for each vertex's vector assignment
    std::vector<std::vector<std::pair<int, int>>> vector_dependencies(k);
    
    // Initialize the first NUM_PREDEFINED_VECTORS with predefined values
    for (int i = 0; i < NUM_PREDEFINED_VECTORS; i++) {
        if (cq_mode) {
            vectors_cq[i] = { predefined_vectors_cq[i][0], predefined_vectors_cq[i][1], predefined_vectors_cq[i][2] };
        } else if (use_complex && i < static_cast<int>(predefined_vectors_complex.size())) {
            for (int j = 0; j < 2 * VECTOR_DIMENSION; j++) vectors[i][j] = predefined_vectors_complex[i][j];
        } else {
            for (int j = 0; j < VECTOR_DIMENSION; j++) vectors[i][j] = predefined_vectors[i][j];
        }
        vector_assigned[i] = true;
        // Predefined vectors have no dependencies
    }
    
    // Debug: Print predefined vectors when orthogonality is enabled
    if (print_statement) {
        DEBUG_PRINT("Predefined vectors for orthogonality checking:");
        for (int i = 0; i < NUM_PREDEFINED_VECTORS; i++) {
            if (cq_mode) {
                DEBUG_PRINT("  v" << i << ": [" << predefined_vectors_cq[i][0].re.a.num << "/" << predefined_vectors_cq[i][0].re.a.den 
                          << "+" << predefined_vectors_cq[i][0].re.b.num << "/" << predefined_vectors_cq[i][0].re.b.den << "*sqrt(" << predefined_vectors_cq[i][0].re.rad << ")"
                          << "+i(" << predefined_vectors_cq[i][0].im.a.num << "/" << predefined_vectors_cq[i][0].im.a.den 
                          << "+" << predefined_vectors_cq[i][0].im.b.num << "/" << predefined_vectors_cq[i][0].im.b.den << "*sqrt(" << predefined_vectors_cq[i][0].im.rad << "))"
                          << ", " << predefined_vectors_cq[i][1].re.a.num << "/" << predefined_vectors_cq[i][1].re.a.den 
                          << "+" << predefined_vectors_cq[i][1].re.b.num << "/" << predefined_vectors_cq[i][1].re.b.den << "*sqrt(" << predefined_vectors_cq[i][1].re.rad << ")"
                          << "+i(" << predefined_vectors_cq[i][1].im.a.num << "/" << predefined_vectors_cq[i][1].im.a.den 
                          << "+" << predefined_vectors_cq[i][1].im.b.num << "/" << predefined_vectors_cq[i][1].im.b.den << "*sqrt(" << predefined_vectors_cq[i][1].im.rad << "))"
                          << ", " << predefined_vectors_cq[i][2].re.a.num << "/" << predefined_vectors_cq[i][2].re.a.den 
                          << "+" << predefined_vectors_cq[i][2].re.b.num << "/" << predefined_vectors_cq[i][2].re.b.den << "*sqrt(" << predefined_vectors_cq[i][2].re.rad << ")"
                          << "+i(" << predefined_vectors_cq[i][2].im.a.num << "/" << predefined_vectors_cq[i][2].im.a.den 
                          << "+" << predefined_vectors_cq[i][2].im.b.num << "/" << predefined_vectors_cq[i][2].im.b.den << "*sqrt(" << predefined_vectors_cq[i][2].im.rad << "))]");
            } else if (use_complex && i < static_cast<int>(predefined_vectors_complex.size())) {
                DEBUG_PRINT("  v" << i << ": [" << predefined_vectors_complex[i][0] << "+" << predefined_vectors_complex[i][1] << "i"
                          << ", " << predefined_vectors_complex[i][2] << "+" << predefined_vectors_complex[i][3] << "i"
                          << ", " << predefined_vectors_complex[i][4] << "+" << predefined_vectors_complex[i][5] << "i]");
            } else {
                DEBUG_PRINT("  v" << i << ": [" << predefined_vectors[i][0] << ", " << predefined_vectors[i][1] << ", " << predefined_vectors[i][2] << "]");
            }
        }
    }
    
    // For each vertex beyond the predefined ones, try to assign vectors
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        
        for (int v = NUM_PREDEFINED_VECTORS; v < k; v++) {
            // Skip if already assigned
            if (vector_assigned[v]) continue;
            
            // Find connected vertices that already have vectors assigned
            std::vector<int> connected_vertices;
            for (int u = 0; u < v; u++) {
                if (matrix[u][v] == l_True && vector_assigned[u]) {
                    connected_vertices.push_back(u);
                    if (connected_vertices.size() >= 2) break;
                }
            }
            
            // If we found at least two connected vertices, we can compute the cross product
            if (connected_vertices.size() >= 2) {
                int u1 = connected_vertices[0];
                int u2 = connected_vertices[1];
                
                if (print_statement) {
                    DEBUG_PRINT("Vertex " << v << " is connected to vertices " << u1 << " and " << u2);
                    DEBUG_PRINT("Vector for vertex " << u1 << ": [" << vectors[u1][0] << "," 
                              << vectors[u1][1] << "," << vectors[u1][2] << "]");
                    DEBUG_PRINT("Vector for vertex " << u2 << ": [" << vectors[u2][0] << "," 
                              << vectors[u2][1] << "," << vectors[u2][2] << "]");
                    
                    // Explain where these vectors came from
                    if (u1 < NUM_PREDEFINED_VECTORS) {
                        DEBUG_PRINT("Vector for vertex " << u1 << " is predefined");
                    } else {
                        DEBUG_PRINT("Vector for vertex " << u1 << " was derived from its connections:");
                        for (const auto& dep : vector_dependencies[u1]) {
                            DEBUG_PRINT("  - Connection to vertex " << dep.second << " (edge " << dep.first << ")");
                        }
                    }
                    
                    if (u2 < NUM_PREDEFINED_VECTORS) {
                        DEBUG_PRINT("Vector for vertex " << u2 << " is predefined");
                    } else {
                        DEBUG_PRINT("Vector for vertex " << u2 << " was derived from its connections:");
                        for (const auto& dep : vector_dependencies[u2]) {
                            DEBUG_PRINT("  - Connection to vertex " << dep.second << " (edge " << dep.first << ")");
                        }
                    }
                    
                    DEBUG_PRINT("Computing cross product to find vector for vertex " << v 
                              << " that is orthogonal to both connected vertices");
                }
                
                // Compute cross product of the two vectors (CQ, complex-int, or real)
                std::vector<int> new_vector;
                std::array<ComplexQuadratic,3> new_vector_cq;
                if (cq_mode) {
                    new_vector_cq = cross_complex_cq(vectors_cq[u1], vectors_cq[u2]);
                } else if (use_complex) {
                    new_vector = cross_complex(vectors[u1], vectors[u2]);
                } else {
                    new_vector = cross_product(vectors[u1], vectors[u2]);
                }
                
                if (print_statement) {
                    DEBUG_PRINT("Cross product result: [" << new_vector[0] << "," 
                              << new_vector[1] << "," << new_vector[2] << "]");
                }
                
                // Check if the cross product is zero (vectors are parallel)
                bool cross_is_zero = false;
                if (cq_mode) {
                    cross_is_zero = cq_is_zero(new_vector_cq[0]) && cq_is_zero(new_vector_cq[1]) && cq_is_zero(new_vector_cq[2]);
                } else if (use_complex) {
                    // Expect interleaved complex: [x_re, x_im, y_re, y_im, z_re, z_im]
                    cross_is_zero = (new_vector.size() == 2 * VECTOR_DIMENSION) &&
                                    (new_vector[0] == 0 && new_vector[1] == 0 &&
                                     new_vector[2] == 0 && new_vector[3] == 0 &&
                                     new_vector[4] == 0 && new_vector[5] == 0);
                } else {
                    cross_is_zero = (new_vector[0] == 0 && new_vector[1] == 0 && new_vector[2] == 0);
                }
                if (cross_is_zero) {
                    if (print_statement) {
                        DEBUG_PRINT("Cross product is zero for vertex " << v);
                        DEBUG_PRINT("Vectors for vertices " << u1 << " and " << u2 << " are parallel or one is zero");
                        DEBUG_PRINT("This means there is no vector that can be orthogonal to both simultaneously");
                        DEBUG_PRINT("Therefore, vertex " << v << " cannot be connected to both " << u1 << " and " << u2);
                    }
                    
                    // Collect all dependencies for this violation
                    std::set<std::pair<int, int>> all_dependencies;
                    
                    // Add direct dependencies
                    int edge_index1 = v*(v-1)/2 + u1;
                    int edge_index2 = v*(v-1)/2 + u2;
                    all_dependencies.insert({edge_index1, u1});
                    all_dependencies.insert({edge_index2, u2});
                    
                    // Add transitive dependencies for u1 and u2
                    for (const auto& dep : vector_dependencies[u1]) {
                        all_dependencies.insert(dep);
                    }
                    for (const auto& dep : vector_dependencies[u2]) {
                        all_dependencies.insert(dep);
                    }
                    
                    if (print_statement) {
                        DEBUG_PRINT("Generated smart blocking clause for parallel vectors with all dependencies:");
                        DEBUG_PRINT("Dependency chain explanation:");
                        
                        for (const auto& dep : all_dependencies) {
                            int edge_var = dep.first;
                            int connected_vertex = dep.second;
                            int vertex1 = 0, vertex2 = 0;
                            
                            // Convert edge variable back to vertex pair
                            for (int j = 1; j < k; j++) {
                                for (int i = 0; i < j; i++) {
                                    int var_idx = j*(j-1)/2 + i;
                                    if (var_idx == edge_var) {
                                        vertex1 = i;
                                        vertex2 = j;
                                        break;
                                    }
                                }
                                if (vertex1 != 0 || vertex2 != 0) break;
                            }
                            
                            DEBUG_PRINT("  Literal " << -(edge_var + 1) << " blocks edge between vertices " 
                                      << vertex1 << " and " << vertex2);
                            
                            if (vertex1 < NUM_PREDEFINED_VECTORS && vertex2 < NUM_PREDEFINED_VECTORS) {
                                DEBUG_PRINT("    Both vertices have predefined vectors");
                            } else if (vertex1 < NUM_PREDEFINED_VECTORS) {
                                DEBUG_PRINT("    Vertex " << vertex1 << " has a predefined vector");
                            } else if (vertex2 < NUM_PREDEFINED_VECTORS) {
                                DEBUG_PRINT("    Vertex " << vertex2 << " has a predefined vector");
                            }
                        }
                        
                        DEBUG_PRINT("This blocking clause prevents configurations where parallel vectors would be required to be orthogonal");
                    }
                    
                    // Create blocking clause from all dependencies
                    for (const auto& dep : all_dependencies) {
                        blocking_clause.push_back(-(dep.first + 1));
                    }
                    
                    orthogonality_violations++;
                    orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
                    return blocking_clause;
                }
                
                // Assign the new vector
                if (cq_mode) vectors_cq[v] = new_vector_cq; else vectors[v] = new_vector;
                vector_assigned[v] = true;
                made_progress = true;
                
                // Record dependencies - FIXED INDEX CALCULATION
                // The edge variable for vertices i and j (where i < j) is calculated as:
                // j*(j-1)/2 + i
                int edge_index1 = v*(v-1)/2 + u1;
                int edge_index2 = v*(v-1)/2 + u2;
                
                if (print_statement) {
                    DEBUG_PRINT("Edge between vertices " << u1 << " and " << v << " has variable index " << edge_index1);
                    DEBUG_PRINT("Edge between vertices " << u2 << " and " << v << " has variable index " << edge_index2);
                }
                
                vector_dependencies[v].push_back({edge_index1, u1});
                vector_dependencies[v].push_back({edge_index2, u2});
                
                // Add transitive dependencies
                for (const auto& dep : vector_dependencies[u1]) {
                    vector_dependencies[v].push_back(dep);
                }
                for (const auto& dep : vector_dependencies[u2]) {
                    vector_dependencies[v].push_back(dep);
                }
                
                if (print_statement) {
                    DEBUG_PRINT("Assigned vector [" << new_vector[0] << "," << new_vector[1] << "," 
                              << new_vector[2] << "] to vertex " << v);
                    DEBUG_PRINT("This vector is orthogonal to both vertex " << u1 << " and vertex " << u2);
                    DEBUG_PRINT("Dependencies for this vector assignment:");
                    DEBUG_PRINT("  - Direct connection to vertex " << u1 << " (edge " << edge_index1 << ")");
                    DEBUG_PRINT("  - Direct connection to vertex " << u2 << " (edge " << edge_index2 << ")");
                    
                    if (!vector_dependencies[u1].empty() || !vector_dependencies[u2].empty()) {
                        DEBUG_PRINT("  - Transitive dependencies:");
                        for (const auto& dep : vector_dependencies[u1]) {
                            DEBUG_PRINT("    - From vertex " << u1 << ": connection to vertex " 
                                      << dep.second << " (edge " << dep.first << ")");
                        }
                        for (const auto& dep : vector_dependencies[u2]) {
                            DEBUG_PRINT("    - From vertex " << u2 << ": connection to vertex " 
                                      << dep.second << " (edge " << dep.first << ")");
                        }
                    }
                    
                    DEBUG_PRINT("Now checking if this vector is orthogonal to all other connected vertices");
                }
                
                // Verify that this vector is orthogonal to all other connected vertices
                bool orthogonality_violation = false;
                int violating_vertex = -1;
                
                for (int u = 0; u < v; u++) {
                    if (matrix[u][v] == l_True && u != u1 && u != u2 && vector_assigned[u]) {
                        int dot = 0;
                        bool dot_zero = true;
                        if (cq_mode) {
                            ComplexQuadratic dcq = dot_complex_cq(vectors_cq[v], vectors_cq[u]);
                            dot_zero = cq_is_zero(dcq);
                        } else if (use_complex) {
                            dot = dot_complex(vectors[v], vectors[u]);
                            dot_zero = (dot == 0);
                        } else {
                            dot = dot_product(vectors[v], vectors[u]);
                            dot_zero = (dot == 0);
                        }
                        if (print_statement) {
                            DEBUG_PRINT("Checking orthogonality with vertex " << u);
                            if (!cq_mode) {
                                DEBUG_PRINT("Vector for vertex " << u << ": [" << vectors[u][0] << "," 
                                          << vectors[u][1] << "," << vectors[u][2] << "]");
                            }
                            if (!cq_mode) DEBUG_PRINT("Dot product: " << dot << " (should be 0 for orthogonal vectors)");
                        }
                        
                        if (!dot_zero) {
                            orthogonality_violation = true;
                            violating_vertex = u;
                            break;
                        }
                    }
                }
                
                if (orthogonality_violation) {
                    if (print_statement) {
                        DEBUG_PRINT("VIOLATION: Vector for vertex " << v << " is not orthogonal to vertex " << violating_vertex);
                        DEBUG_PRINT("This means vertex " << v << " cannot be connected to vertices " 
                                  << u1 << ", " << u2 << ", and " << violating_vertex << " simultaneously");
                        DEBUG_PRINT("The vector for vertex " << v << " was determined by its connections to vertices " 
                                  << u1 << " and " << u2);
                        DEBUG_PRINT("But this vector is not orthogonal to vertex " << violating_vertex);
                    }
                    
                    // Collect all dependencies for this violation
                    std::set<std::pair<int, int>> all_dependencies;
                    
                    // Add direct dependencies
                    int var_index1 = v*(v-1)/2 + u1;
                    int var_index2 = v*(v-1)/2 + u2;
                    int var_index3 = v*(v-1)/2 + violating_vertex;
                    all_dependencies.insert({var_index1, u1});
                    all_dependencies.insert({var_index2, u2});
                    all_dependencies.insert({var_index3, violating_vertex});
                    
                    // Add transitive dependencies for u1, u2, and violating_vertex
                    for (const auto& dep : vector_dependencies[u1]) {
                        all_dependencies.insert(dep);
                    }
                    for (const auto& dep : vector_dependencies[u2]) {
                        all_dependencies.insert(dep);
                    }
                    for (const auto& dep : vector_dependencies[violating_vertex]) {
                        all_dependencies.insert(dep);
                    }
                    
                    if (print_statement) {
                        DEBUG_PRINT("Generated smart blocking clause for non-orthogonal vectors with all dependencies:");
                        DEBUG_PRINT("Dependency chain explanation:");
                        
                        for (const auto& dep : all_dependencies) {
                            int edge_var = dep.first;
                            int connected_vertex = dep.second;
                            int vertex1 = 0, vertex2 = 0;
                            
                            // Convert edge variable back to vertex pair
                            for (int j = 1; j < k; j++) {
                                for (int i = 0; i < j; i++) {
                                    int var_idx = j*(j-1)/2 + i;
                                    if (var_idx == edge_var) {
                                        vertex1 = i;
                                        vertex2 = j;
                                        break;
                                    }
                                }
                                if (vertex1 != 0 || vertex2 != 0) break;
                            }
                            
                            DEBUG_PRINT("  Literal " << -(edge_var + 1) << " blocks edge between vertices " 
                                      << vertex1 << " and " << vertex2);
                            
                            if (edge_var == var_index1) {
                                DEBUG_PRINT("    This edge directly determines the vector for vertex " << v);
                            } else if (edge_var == var_index2) {
                                DEBUG_PRINT("    This edge directly determines the vector for vertex " << v);
                            } else if (edge_var == var_index3) {
                                DEBUG_PRINT("    This edge creates the orthogonality violation with vertex " << violating_vertex);
                            } else {
                                DEBUG_PRINT("    This edge is part of the dependency chain for determining vectors");
                            }
                        }
                        
                        DEBUG_PRINT("This blocking clause prevents configurations where non-orthogonal vectors would be connected");
                    }
                    
                    // Create blocking clause from all dependencies
                    for (const auto& dep : all_dependencies) {
                        blocking_clause.push_back(-(dep.first + 1));
                    }

                    // Write orthogonality clause as trusted clause to DRAT proof (like cadical-ks)
                    solver->add_trusted_orthogonality_clause(blocking_clause);

                    // Write witness to .ortho file
                    write_orthogonality_witness(blocking_clause, vectors, vectors_cq, cq_mode);

                    orthogonality_violations++;
                    orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
                    return blocking_clause;
                }
            }
        }
    }

    // Now check if all edges satisfy orthogonality
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < j; i++) {
            // Skip checking edges between predefined vertices
            if (i < NUM_PREDEFINED_VECTORS && j < NUM_PREDEFINED_VECTORS) {
                continue;
            }
            
            // Skip if either vector is not assigned or if the edge is undefined
            if (!vector_assigned[i] || !vector_assigned[j] || matrix[i][j] == l_Undef) {
                continue;
            }
            
            int dot = 0; bool dot_zero = true;
            if (cq_mode) {
                ComplexQuadratic dcq = dot_complex_cq(vectors_cq[i], vectors_cq[j]);
                dot_zero = cq_is_zero(dcq);
            } else if (use_complex) {
                dot = dot_complex(vectors[i], vectors[j]);
                dot_zero = (dot == 0);
            } else {
                dot = dot_product(vectors[i], vectors[j]);
                dot_zero = (dot == 0);
            }
            
            if (print_statement) {
                DEBUG_PRINT("Checking direct orthogonality between vertices " << i << " and " << j);
                DEBUG_PRINT("Vector for vertex " << i << ": [" << vectors[i][0] << "," 
                          << vectors[i][1] << "," << vectors[i][2] << "]");
                DEBUG_PRINT("Vector for vertex " << j << ": [" << vectors[j][0] << "," 
                          << vectors[j][1] << "," << vectors[j][2] << "]");
                DEBUG_PRINT("Dot product: " << dot << " (should be 0 for orthogonal vectors)");
                DEBUG_PRINT("Edge exists between these vertices: " << (matrix[i][j] == l_True ? "Yes" : "No"));
            }
            
            // Only check for the case where vectors are not orthogonal but they are connected
            if (!dot_zero && matrix[i][j] == l_True) {
                if (print_statement) {
                    DEBUG_PRINT("VIOLATION: Vertices " << i << " and " << j << " are connected but their vectors are not orthogonal");
                    DEBUG_PRINT("Connected vertices must have orthogonal vectors (dot product = 0)");
                    if (!cq_mode) DEBUG_PRINT("But the dot product is " << dot << " ≠ 0");
                }
                
                // Collect all dependencies for this violation
                std::set<std::pair<int, int>> all_dependencies;
                
                // Add direct dependency
                int var_index = j*(j-1)/2 + i;
                all_dependencies.insert({var_index, i});
                
                // Add transitive dependencies for i and j
                for (const auto& dep : vector_dependencies[i]) {
                    all_dependencies.insert(dep);
                }
                for (const auto& dep : vector_dependencies[j]) {
                    all_dependencies.insert(dep);
                }
                
                if (print_statement) {
                    DEBUG_PRINT("Generated smart blocking clause for direct orthogonality violation with all dependencies:");
                    DEBUG_PRINT("Dependency chain explanation:");
                    
                    for (const auto& dep : all_dependencies) {
                        int edge_var = dep.first;
                        int connected_vertex = dep.second;
                        int vertex1 = 0, vertex2 = 0;
                        
                        // Convert edge variable back to vertex pair
                        for (int j2 = 1; j2 < k; j2++) {
                            for (int i2 = 0; i2 < j2; i2++) {
                                int var_idx = j2*(j2-1)/2 + i2;
                                if (var_idx == edge_var) {
                                    vertex1 = i2;
                                    vertex2 = j2;
                                    break;
                                }
                            }
                            if (vertex1 != 0 || vertex2 != 0) break;
                        }
                        
                        DEBUG_PRINT("  Literal " << -(edge_var + 1) << " blocks edge between vertices " 
                                  << vertex1 << " and " << vertex2);
                        
                        if (edge_var == var_index) {
                            DEBUG_PRINT("    This is the direct edge that violates orthogonality");
                        } else {
                            DEBUG_PRINT("    This edge is part of the dependency chain for determining vectors");
                        }
                    }
                    
                    DEBUG_PRINT("This blocking clause prevents configurations where non-orthogonal vectors would be connected");
                }
                
                // Create blocking clause from all dependencies
                for (const auto& dep : all_dependencies) {
                    blocking_clause.push_back(-(dep.first + 1));
                }

                // Write orthogonality clause as trusted clause to DRAT proof (like cadical-ks)
                solver->add_trusted_orthogonality_clause(blocking_clause);

                // Write witness to .ortho file
                write_orthogonality_witness(blocking_clause, vectors, vectors_cq, cq_mode);

                orthogonality_violations++;
                orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
                return blocking_clause;
            }
        }
    }
    
    orthogonality_check_time += CaDiCaL::absolute_process_time() - start_time;
    return blocking_clause;
}

// Modify the block_extension function to conditionally check orthogonality
std::vector<int> SymmetryBreaker::block_extension(int k) {
    std::string input = convert_assignment_to_string(k);
    if (print_statement) {
        std::cout << "Checking for partial assignment: " << input << std::endl;
        std::cout << "DEBUG: use_master_graph = " << use_master_graph << std::endl;
        std::cout << "DEBUG: use_orthogonality = " << use_orthogonality << std::endl;
    }
    
    // Convert to adjacency matrix
    adjacency_matrix_t matrix = string_to_adjacency_matrix(input, k);
    
    // Check if the partial assignment is a subgraph of the master graph
    if (use_master_graph) {
        bool is_subgraph = is_subgraph_of_master(matrix);
        if (is_subgraph) {
            subgraph_count++;
        } else {
            non_subgraph_count++;
            if (print_statement) {
                std::cout << "DEBUG: Partial assignment is not a subgraph of the master graph" << std::endl;
            }
            return generate_naive_blocking_clause(input, true);
        }
    }
    
    // Check orthogonality if enabled
    if (use_orthogonality && k > NUM_PREDEFINED_VECTORS) {
        std::vector<int> orthogonality_clause = check_orthogonality_constraints(input, k);
        if (!orthogonality_clause.empty()) {
            if (print_statement) {
                std::cout << "DEBUG: Orthogonality constraint violation detected" << std::endl;
            }
            return orthogonality_clause;
        }
    }
    
    const double before = CaDiCaL::absolute_process_time();
    std::vector<int> blocking_clause = call_RCL_binary(input, k);
    const double after = CaDiCaL::absolute_process_time();
    
    if (blocking_clause.empty()) {
        canon++;
        canontime += (after-before);
        canonarr[k-1]++;
        canontimearr[k-1] += (after-before);
    } else {
        noncanon++;
        noncanontime += (after-before);
        noncanonarr[k-1]++;
        noncanontimearr[k-1] += (after-before);
    }

    if (!blocking_clause.empty()) {
        learned_clauses_count++;
    }

    return blocking_clause;
}

void SymmetryBreaker::notify_assignment(int lit, bool is_fixed) {
    int var = std::abs(lit) - 1;
    if (var < num_edge_vars) {
        assign[var] = (lit > 0) ? l_True : l_False;
        fixed[var] = is_fixed;
    }
}

void SymmetryBreaker::notify_new_decision_level() {
    current_trail.push_back(std::vector<int>());
}

void SymmetryBreaker::notify_backtrack(size_t new_level) {
    while (current_trail.size() > new_level) {
        for (int lit : current_trail.back()) {
            int var = std::abs(lit) - 1;
            if (var < num_edge_vars) {
                assign[var] = l_Undef;
                fixed[var] = false;
            }
        }
        current_trail.pop_back();
    }
}

bool SymmetryBreaker::cb_check_found_model(const std::vector<int>& model) {
    std::vector<int> blocking_clause = block_extension(n);

    if (blocking_clause.empty()) {  // Canonical
        sol_count++;
        
        // Optimize: avoid string generation when possible
        if (print_statement && !no_print) {
            // Both debug and solution output needed - generate string once
            std::string full_assignment = convert_assignment_to_string(n);
            std::cout << "Found canonical solution #" << sol_count << ": " << full_assignment << std::endl;
            printf("Solution %ld: %s\n", sol_count, full_assignment.c_str());
            fflush(stdout);
        } else if (print_statement) {
            // Only debug output needed
            std::string full_assignment = convert_assignment_to_string(n);
            std::cout << "Found canonical solution #" << sol_count << ": " << full_assignment << std::endl;
        } else if (!no_print) {
            // Only solution output needed - use efficient direct printing
            print_solution_direct(sol_count);
        }
        
        // Generate a blocking clause for this solution
        for (int i = 0; i < num_edge_vars; i++) {
            blocking_clause.push_back(assign[i] == l_True ? -(i + 1) : (i + 1));
        }
    } else if (print_statement) {
        // Only generate string for debug output when needed
        std::string full_assignment = convert_assignment_to_string(n);
        std::cout << "Found non-canonical full assignment: " << full_assignment << std::endl;
    }

    new_clauses.push_back(blocking_clause);
    
    // Trace solution blocking clause to DRAT proof (like cadical-ks)
    solver->add_trusted_clause(blocking_clause);
    
    return false;
}

bool SymmetryBreaker::cb_has_external_clause() {
    if (!new_clauses.empty()) {
        if (print_statement) {
            DEBUG_PRINT("Found existing clauses in queue, returning true");
        }
        return true;
    }

    static int subgraph_check_counter = 0;
    static std::unordered_set<std::string> seen_subgraph_assignments;
    
    // Check if we only have variables <= 78 assigned
    bool only_small_vars = true;
    int highest_assigned_var = 0;
    
    for (int var = 0; var < num_edge_vars; var++) {
        if (assign[var] != l_Undef) {
            highest_assigned_var = std::max(highest_assigned_var, var + 1);
            if (var + 1 > EDGE_CUTOFF) {
                only_small_vars = false;
                break;
            }
        }
    }
    
    // Skip all external propagator checks if we only have small variables
    if (only_small_vars) {
        if (print_statement) {
            DEBUG_PRINT("Skipping external propagator checks - only variables <= 78 are assigned (highest: " 
                      << highest_assigned_var << ")");
        }
        return false;
    }
    
    // Print current partial assignment
    std::string current_assignment_str = convert_assignment_to_string(n);
    if (print_statement) {
        DEBUG_PRINT("Processing partial assignment: " << current_assignment_str);
    }
    
    // First check orthogonality for any partial assignment with at least 14 vertices
    // This is separate from the canonicity check which requires complete assignments
    if (use_orthogonality && n > NUM_PREDEFINED_VECTORS) {
        // Check if we have any edges involving vertices > 13
        bool has_large_vertex_edges = false;
        for (int j = NUM_PREDEFINED_VECTORS; j < n; j++) {
            for (int i = 0; i < j; i++) {
                int var_index = j*(j-1)/2 + i;
                if (assign[var_index] == l_True) {
                    has_large_vertex_edges = true;
                    break;
                }
            }
            if (has_large_vertex_edges) break;
        }
        
        if (has_large_vertex_edges) {
            // We have some edges involving vertices > 13, check orthogonality
            std::vector<int> orthogonality_clause = check_orthogonality_constraints(current_assignment_str, n);
            if (!orthogonality_clause.empty()) {
                if (print_statement) {
                    DEBUG_PRINT("Orthogonality constraint violation detected for partial assignment");
                    DEBUG_PRINT("Generated orthogonality blocking clause: ");
                    for (int lit : orthogonality_clause) {
                        std::cout << "  " << lit;
                    }
                    std::cout << std::endl;
                }
                
                new_clauses.push_back(orthogonality_clause);
                return true;
            }
        }
    }
    
    // Then check if current partial assignment is a subgraph of master graph
    if (use_master_graph && subgraph_check_counter % 5 == 0) {
        if (print_statement) {
            DEBUG_PRINT("Starting master graph subgraph check");
        }
        
        // Create a string representation of the current assignment (only true edges)
        std::string current_assignment;
        current_assignment.reserve(num_edge_vars * 4); // Pre-allocate memory
        
        // Quick check if we have any true assignments
        bool has_true_assignments = false;
        for (int j = 1; j < n && !has_true_assignments; j++) {
            for (int i = 0; i < j; i++) {
                int var_index = j*(j-1)/2 + i;
                if (assign[var_index] == l_True) {
                    has_true_assignments = true;
                    break;
                }
            }
        }
        
        // If no true assignments, it's trivially a subgraph
        bool need_to_check = has_true_assignments;
        
        if (need_to_check) {
            for (int j = 1; j < n; j++) {
                for (int i = 0; i < j; i++) {
                    int var_index = j*(j-1)/2 + i;
                    if (assign[var_index] == l_True) {
                        current_assignment += std::to_string(var_index) + ",";
                    }
                }
            }
            
            // Check if we've already processed this assignment
            if (seen_subgraph_assignments.find(current_assignment) != seen_subgraph_assignments.end()) {
                need_to_check = false;
            } else {
                seen_subgraph_assignments.insert(current_assignment);
            }
        }
        
        if (need_to_check) {
            if (print_statement) {
                DEBUG_PRINT("Converting partial assignment to adjacency matrix");
            }
            adjacency_matrix_t partial_matrix = string_to_adjacency_matrix(current_assignment_str, n);
            
            if (print_statement) {
                DEBUG_PRINT("Checking if partial assignment is subgraph of master graph");
            }
            bool is_subgraph = is_subgraph_of_master(partial_matrix);
            
            if (!is_subgraph) {
                if (print_statement) {
                    DEBUG_PRINT("Partial assignment is NOT a subgraph of master graph");
                }
                std::vector<int> blocking_clause = generate_naive_blocking_clause(current_assignment_str, true);
                
                if (!blocking_clause.empty() && print_statement) {
                    DEBUG_PRINT("Generated subgraph blocking clause: ");
                    for (int lit : blocking_clause) {
                        std::cout << "  " << lit;
                    }
                    std::cout << std::endl;
                }
                
                new_clauses.push_back(blocking_clause);
                non_subgraph_count++;
                return true;
            } else if (print_statement) {
                DEBUG_PRINT("Partial assignment IS a subgraph of master graph");
            }
        }
    }
    
    subgraph_check_counter++;

    // Then proceed with the regular canonicity checks (requiring complete assignments)
    for (int k = SMALL_GRAPH_ORDER + 1; k <= n; k++) {
        bool is_complete = true;
        for (int j = 0; j < k*(k-1)/2; j++) {
            if (assign[j] == l_Undef) {
                is_complete = false;
                break;
            }
        }
        
        if (is_complete) {
            std::string partial_assignment = convert_assignment_to_string(k);
            
            
            // Skip orthogonality check here since we already did it above for any partial assignment
            
            // Only proceed with canonicity check
            size_t hash_value = std::hash<std::string>{}(partial_assignment);

            if (seen_partial_assignments.find(hash_value) != seen_partial_assignments.end()) {
                if (print_statement) {
                    DEBUG_PRINT("Skipping already processed partial assignment of size " << k);
                }
                continue;
            }

            seen_partial_assignments.insert(hash_value);
            if (print_statement) {
                DEBUG_PRINT("Starting canonicity check for partial assignment of size " << k);
            }

            std::vector<int> blocking_clause = block_extension(k);
            
            if (!blocking_clause.empty()) {
                if (print_statement) {
                    DEBUG_PRINT("Partial assignment of size " << k << " is non-canonical");
                    DEBUG_PRINT("Generated blocking clause: ");
                    for (int lit : blocking_clause) {
                        std::cout << "  " << lit;
                    }
                    std::cout << std::endl;
                }
                
                new_clauses.push_back(blocking_clause);
                return true;
            } else {
                if (print_statement) {
                    DEBUG_PRINT("Partial assignment of size " << k << " is canonical");
                }
            }
        } else if (print_statement) {
            DEBUG_PRINT("Partial assignment of size " << k << " is incomplete, stopping canonicity checks");
            break;
        }
    }

    if (print_statement) {
        DEBUG_PRINT("No clauses generated for current partial assignment");
    }
    return false;
}

int SymmetryBreaker::cb_add_external_clause_lit () {
    if (new_clauses.empty()) return 0;
    else {
        assert(!new_clauses.empty());
        size_t clause_idx = new_clauses.size() - 1;
        if (new_clauses[clause_idx].empty()) {
            new_clauses.pop_back();
            return 0;
        }

        int lit = new_clauses[clause_idx].back();
        new_clauses[clause_idx].pop_back();
        return lit;
    }
}

int SymmetryBreaker::cb_decide() {
    return 0;
}

int SymmetryBreaker::cb_propagate() {
    return 0;
}

int SymmetryBreaker::cb_add_reason_clause_lit(int plit) {
    (void)plit; // Suppress unused parameter warning
    return 0;
}

void SymmetryBreaker::printGraph(const graph* g, int n, int m) {
    if (print_statement) {
        std::cout << "Graph representation:" << std::endl;
        for (int i = 0; i < n; i++) {
            std::cout << "  Node " << i << " connected to: ";
            for (int j = 0; j < n; j++) {
                if (ISELEMENT(GRAPHROW(g, i, m), j)) {
                    std::cout << j << " ";
                }
            }
            std::cout << std::endl;
        }
    }
}

void SymmetryBreaker::initNautyOptions() {
    options.writeautoms = FALSE;
    options.writemarkers = FALSE;
    options.getcanon = TRUE;
    options.defaultptn = TRUE;
}

// Modify the generate_complex_blocking_clause function to remove the degree check
std::vector<int> SymmetryBreaker::generate_complex_blocking_clause(const std::string& assignment, int k) {
    std::vector<int> blocking_clause;
    
    // Find the minimal noncanonical matrix size
    int minimal_k = k;
    std::string current_assignment = assignment;
    
    while (minimal_k > 1) {
        std::string submatrix = extract_submatrix(current_assignment, minimal_k - 1);
        if (isCanonical(submatrix)) {
            break;
        }
        current_assignment = submatrix;
        minimal_k--;
    }
    
    if (print_statement) {
        std::cout << "DEBUG: Minimal noncanonical matrix size: " << minimal_k << std::endl;
        std::cout << "DEBUG: Minimal noncanonical assignment: " << current_assignment << std::endl;
    }
    
    if (minimal_k > 1) {
        for (size_t i = 0; i < current_assignment.length(); ++i) {
            int literal = current_assignment[i] == '1' ? -(i + 1) : (i + 1);
            blocking_clause.push_back(literal);
        }
    }

    if (print_statement) {
        std::cout << "DEBUG: Generated blocking clause: ";
        for (int lit : blocking_clause) {
            std::cout << lit << " ";
        }
        std::cout << std::endl;
    }

    return blocking_clause;
}

// Optimize the generate_naive_blocking_clause function to avoid unnecessary string operations
std::vector<int> SymmetryBreaker::generate_naive_blocking_clause(const std::string& assignment, bool only_true_edges) {
    std::vector<int> blocking_clause;
    blocking_clause.reserve(assignment.length()); // Pre-allocate memory
    
    // Skip the first 78 variables (indices 0-77)
    const int skip_vars = EDGE_CUTOFF;
    
    for (size_t i = 0; i < assignment.length(); ++i) {
        // Only include variables with index > 78
        if (i + 1 > skip_vars) {
            if (only_true_edges) {
                // For subgraph blocking, only include TRUE edges
                if (assignment[i] == '1') {
                    blocking_clause.push_back(-(i + 1));
                }
            } else {
                // For canonicity blocking, include both TRUE and FALSE edges
                blocking_clause.push_back(assignment[i] == '1' ? -(i + 1) : (i + 1));
            }
        }
    }

    if (print_statement) {
        std::cout << "DEBUG: Generated " << (only_true_edges ? "subgraph" : "canonicity") 
                  << " blocking clause (skipping first " << skip_vars << " variables): ";
        for (int lit : blocking_clause) {
            std::cout << lit << " ";
        }
        std::cout << std::endl;
    }

    return blocking_clause;
}

// Optimize the string_to_adjacency_matrix function
adjacency_matrix_t SymmetryBreaker::string_to_adjacency_matrix(const std::string& input, int k) {
    adjacency_matrix_t matrix(k, std::vector<truth_value_t>(k, l_False));
    int index = 0;
    
    for (int j = 1; j < k; j++) {
        for (int i = 0; i < j; i++) {
            if (index < input.length() && input[index++] == '1') {
                matrix[i][j] = l_True;
                matrix[j][i] = l_True;
            }
        }
    }
    return matrix;
}

std::string SymmetryBreaker::adjacency_matrix_to_string(const adjacency_matrix_t& matrix) {
    std::string result;
    int k = matrix.size();
    for (int j = 1; j < k; j++) {
        for (int i = 0; i < j; i++) {
            result += (matrix[i][j] == l_True) ? '1' : '0';  // Changed to l_True
        }
    }
    return result;
}

void SymmetryBreaker::load_master_graph(const std::string& filepath) {
    // Set print_statement to true at the beginning of this function
    bool original_print_setting = print_statement;
    print_statement = true;
    
    // Check if we've already loaded the master graph
    if (!masterGraph.empty()) {
        std::cout << "DEBUG: Master graph already loaded, skipping reload" << std::endl;
        print_statement = original_print_setting;  // Restore original setting
        return;
    }
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open master graph file " << filepath << std::endl;
        std::cerr << "Please check that the file exists and is readable" << std::endl;
        print_statement = original_print_setting;  // Restore original setting
        return;
    }

    // Basic implementation for .lad format
    std::string line;
    std::getline(file, line); // Read the first line (number of vertices)
    int n = std::stoi(line);
    masterGraph = adjacency_matrix_t(n, std::vector<truth_value_t>(n, l_False));
    
    // Store vertex labels
    std::vector<int> vertexLabels(n);

    for (int i = 0; i < n; i++) {
        std::getline(file, line);
        std::istringstream iss(line);
        
        int label;
        iss >> label;  // Read the first number as the vertex label
        vertexLabels[i] = label;
        
        int degree;
        iss >> degree;  // Read the number of successors
        
        for (int j = 0; j < degree; j++) {
            int neighbor;
            iss >> neighbor;
            masterGraph[i][neighbor] = l_True;
            masterGraph[neighbor][i] = l_True;  // For undirected graph
        }
    }

    if (print_statement) {
        std::cout << "DEBUG: Loaded master graph with " << n << " vertices" << std::endl;
        std::cout << "DEBUG: Vertex labels: ";
        for (int i = 0; i < n; i++) {
            std::cout << vertexLabels[i] << " ";
        }
        std::cout << std::endl;
        
        // Print out connections for each vertex
        // std::cout << "DEBUG: Master graph connections:" << std::endl;
        // for (int i = 0; i < n; i++) {
        //     std::cout << "Vertex " << i << " (label " << vertexLabels[i] << ") connects to: ";
        //     bool first = true;
        //     for (int j = 0; j < n; j++) {
        //         if (masterGraph[i][j] == l_True) {
        //             if (!first) {
        //                 std::cout << ", ";
        //             }
        //             std::cout << j;
        //             first = false;
        //         }
        //     }
        //     std::cout << std::endl;
        // }
    }
    
    // Store vertex labels as a member variable for later use
    masterGraphLabels = vertexLabels;
    use_master_graph = true;
    
    // Restore original print_statement setting at the end
    print_statement = original_print_setting;
}

bool SymmetryBreaker::is_subgraph_of_master(const adjacency_matrix_t& graph) {
    // Add timer start
    const double before = CaDiCaL::absolute_process_time();

    // Implement a simple subgraph check without Glasgow
    if (masterGraph.empty()) {
        std::cerr << "Error: Master graph not loaded" << std::endl;
        const double after = CaDiCaL::absolute_process_time();
        subgraph_check_time += (after - before);
        return false;
    }

    // Simple subgraph check: for each edge in the input graph, check if it exists in the master graph
    int n = graph.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (graph[i][j] == l_True) {
                // If this edge exists in the input graph, it must exist in the master graph
                if (i >= masterGraph.size() || j >= masterGraph.size() || masterGraph[i][j] != l_True) {
                    const double after = CaDiCaL::absolute_process_time();
                    subgraph_check_time += (after - before);
                    return false;
                }
            }
        }
    }

    const double after = CaDiCaL::absolute_process_time();
    subgraph_check_time += (after - before);
    return true;
}

void SymmetryBreaker::print_subgraph_statistics() {
    if (use_master_graph) {
        printf("Subgraph checks       : %-12d\n", subgraph_count + non_subgraph_count);
        printf("  Subgraphs           : %-12d\n", subgraph_count);
        printf("  Non-subgraphs       : %-12d\n", non_subgraph_count);
        printf("Subgraph check time   : %g s\n", subgraph_check_time);
    }
    
    if (use_orthogonality) {
        printf("Orthogonality checks  : %-12ld\n", orthogonality_checks);
        printf("  Skipped checks      : %-12ld\n", orthogonality_skipped);
        printf("  Violations found    : %-12ld\n", orthogonality_violations);
        printf("Orthogonality time    : %g s\n", orthogonality_check_time);
    }
}

// Add this setter method
void SymmetryBreaker::set_use_orthogonality(bool value) {
    use_orthogonality = value;
    if (print_statement) {
        DEBUG_PRINT("Orthogonality checking " << (value ? "enabled" : "disabled"));
    }
}

void SymmetryBreaker::print_orthogonality_status() {
    // Print orthogonality configuration status
    std::cout << "c Orthogonality configuration:" << std::endl;
    std::cout << "c   enabled: " << (use_orthogonality ? "true" : "false") << std::endl;
    std::cout << "c   complex: " << (use_complex ? "true" : "false") << std::endl;
    
    // Print initial vectors information
    if (use_complex) {
        if (!predefined_vectors_cq.empty()) {
            std::cout << "c   initial vectors: " << predefined_vectors_cq.size() << " complex vectors with exact sqrt support" << std::endl;
            for (size_t i = 0; i < predefined_vectors_cq.size(); i++) {
                std::cout << "c     v" << (i+1) << ": [" 
                          << predefined_vectors_cq[i][0].re.a.num << "/" << predefined_vectors_cq[i][0].re.a.den
                          << "+" << predefined_vectors_cq[i][0].re.b.num << "/" << predefined_vectors_cq[i][0].re.b.den << "*sqrt(" << predefined_vectors_cq[i][0].re.rad << ")"
                          << "+i(" << predefined_vectors_cq[i][0].im.a.num << "/" << predefined_vectors_cq[i][0].im.a.den 
                          << "+" << predefined_vectors_cq[i][0].im.b.num << "/" << predefined_vectors_cq[i][0].im.b.den << "*sqrt(" << predefined_vectors_cq[i][0].im.rad << "))"
                          << ", "
                          << predefined_vectors_cq[i][1].re.a.num << "/" << predefined_vectors_cq[i][1].re.a.den
                          << "+" << predefined_vectors_cq[i][1].re.b.num << "/" << predefined_vectors_cq[i][1].re.b.den << "*sqrt(" << predefined_vectors_cq[i][1].re.rad << ")"
                          << "+i(" << predefined_vectors_cq[i][1].im.a.num << "/" << predefined_vectors_cq[i][1].im.a.den 
                          << "+" << predefined_vectors_cq[i][1].im.b.num << "/" << predefined_vectors_cq[i][1].im.b.den << "*sqrt(" << predefined_vectors_cq[i][1].im.rad << "))"
                          << ", "
                          << predefined_vectors_cq[i][2].re.a.num << "/" << predefined_vectors_cq[i][2].re.a.den
                          << "+" << predefined_vectors_cq[i][2].re.b.num << "/" << predefined_vectors_cq[i][2].re.b.den << "*sqrt(" << predefined_vectors_cq[i][2].re.rad << ")"
                          << "+i(" << predefined_vectors_cq[i][2].im.a.num << "/" << predefined_vectors_cq[i][2].im.a.den 
                          << "+" << predefined_vectors_cq[i][2].im.b.num << "/" << predefined_vectors_cq[i][2].im.b.den << "*sqrt(" << predefined_vectors_cq[i][2].im.rad << "))]"
                          << std::endl;
            }
        } else if (!predefined_vectors_complex.empty()) {
            std::cout << "c   initial vectors: " << predefined_vectors_complex.size() << " complex vectors" << std::endl;
            for (size_t i = 0; i < predefined_vectors_complex.size(); i++) {
                std::cout << "c     v" << (i+1) << ": [" 
                          << predefined_vectors_complex[i][0] << "+" << predefined_vectors_complex[i][1] << "i"
                          << ", " << predefined_vectors_complex[i][2] << "+" << predefined_vectors_complex[i][3] << "i"
                          << ", " << predefined_vectors_complex[i][4] << "+" << predefined_vectors_complex[i][5] << "i]" << std::endl;
            }
        } else {
            std::cout << "c   initial vectors: none (will use real vectors)" << std::endl;
        }
    } else {
        std::cout << "c   initial vectors: " << predefined_vectors.size() << " real vectors" << std::endl;
        for (size_t i = 0; i < predefined_vectors.size(); i++) {
            std::cout << "c     v" << (i+1) << ": [" 
                      << predefined_vectors[i][0] << ", " 
                      << predefined_vectors[i][1] << ", " 
                      << predefined_vectors[i][2] << "]" << std::endl;
        }
    }
}

// Add this setter method for controlling solution printing
void SymmetryBreaker::set_no_print(bool value) {
    no_print = value;
    if (print_statement) {
        DEBUG_PRINT("Solution printing " << (value ? "disabled" : "enabled"));
    }
}

// Efficient solution printing without string allocation
void SymmetryBreaker::print_solution_direct(long solution_number) {
    printf("Solution %ld: ", solution_number);
    
    // Print directly without creating intermediate string
    for (int j = 1; j < n; j++) {
        for (int i = 0; i < j; i++) {
            putchar((assign[j*(j-1)/2 + i] == l_True) ? '1' : '0');
        }
    }
    putchar('\n');
    fflush(stdout);
}

void SymmetryBreaker::set_permutation_filename(const std::string& filename) {
    perm_filename = filename;
    if (print_statement) {
        DEBUG_PRINT("Permutation filename set to: " << filename);
    }
}

void SymmetryBreaker::set_orthogonality_filename(const std::string& filename) {
    ortho_filename = filename;
    if (print_statement) {
        DEBUG_PRINT("Orthogonality filename set to: " << filename);
    }
}

void SymmetryBreaker::set_binary_format(bool value) {
    binary_format = value;
    if (print_statement) {
        DEBUG_PRINT("Binary format " << (value ? "enabled" : "disabled"));
    }
}

void SymmetryBreaker::write_permutation_to_file(const std::vector<int>& permutation) {
    // Initialize permutation file if not already done
    if (!perm_file && !perm_filename.empty()) {
        perm_file = fopen(perm_filename.c_str(), binary_format ? "wb" : "w");
        if (!perm_file) {
            std::cerr << "Error: Could not open permutation file " << perm_filename << std::endl;
            return;
        }
    }

    if (perm_file) {
        if (binary_format) {
            // Write in binary format using variable-length encoding
            // Each integer is encoded as variable-length: (abs(val)+1)*2 + sign bit
            // This ensures 0 is never encoded as 0x00 (which is the terminator)
            for (size_t i = 0; i < permutation.size(); i++) {
                int val = permutation[i];
                // Encode as: x = 2 * (abs(val) + 1) + (val < 0)
                // This shifts all values up by 1 to avoid 0x00 encoding
                unsigned int x = 2u * (abs(val) + 1) + (val < 0 ? 1 : 0);
                
                // Write variable-length encoded integer
                while (x & ~0x7f) {
                    unsigned char ch = (x & 0x7f) | 0x80;
                    fwrite(&ch, 1, 1, perm_file);
                    x >>= 7;
                }
                unsigned char ch = x;
                fwrite(&ch, 1, 1, perm_file);
            }
            // Write terminator (0 byte)
            unsigned char zero = 0;
            fwrite(&zero, 1, 1, perm_file);
        } else {
            // Write in text format with spaces between numbers
            for (size_t i = 0; i < permutation.size(); i++) {
                if (i > 0) fprintf(perm_file, " ");
                fprintf(perm_file, "%d", permutation[i]);
            }
            fprintf(perm_file, "\n");
        }
        fflush(perm_file);

        if (print_statement) {
            DEBUG_PRINT("Wrote permutation to file (" << (binary_format ? "binary" : "text") << "): ");
            for (size_t i = 0; i < permutation.size(); i++) {
                if (i > 0) std::cout << " ";
                std::cout << permutation[i];
            }
            std::cout << std::endl;
        }
    }
}

void SymmetryBreaker::write_orthogonality_witness(const std::vector<int>& blocking_clause,
                                                   const std::vector<std::vector<int>>& vectors_int,
                                                   const std::vector<std::array<ComplexQuadratic,3>>& vectors_cq_param,
                                                   bool use_cq) {
    // Initialize orthogonality file if not already done
    if (!ortho_file && !ortho_filename.empty()) {
        ortho_file = fopen(ortho_filename.c_str(), "w");
        if (!ortho_file) {
            std::cerr << "Error: Could not open orthogonality file " << ortho_filename << std::endl;
            return;
        }
    }

    if (ortho_file) {
        // Write edges from the blocking clause
        fprintf(ortho_file, "edges:");
        for (size_t i = 0; i < blocking_clause.size(); i++) {
            if (i > 0) fprintf(ortho_file, ",");
            // blocking_clause contains negative literals, convert back to edge variable
            int edge_var = -blocking_clause[i] - 1;
            fprintf(ortho_file, "%d", edge_var);
        }

        // Write vectors for vertices that appear in the blocking clause
        fprintf(ortho_file, " vectors:");

        // Extract unique vertices from the edges in the blocking clause
        std::set<int> vertices_set;
        for (int lit : blocking_clause) {
            int edge_var = -lit - 1;
            // Convert edge variable to vertex pair (i, j)
            // edge_var = j*(j-1)/2 + i
            int j = 1;
            while (j * (j - 1) / 2 <= edge_var) j++;
            j--;
            int i = edge_var - j * (j - 1) / 2;
            vertices_set.insert(i);
            vertices_set.insert(j);
        }

        // Write vectors in order of vertices
        bool first = true;
        for (int v : vertices_set) {
            if (!first) fprintf(ortho_file, ",");
            first = false;

            fprintf(ortho_file, "v%d=[", v);
            if (use_cq) {
                // Write complex quadratic vectors
                const auto& vec = vectors_cq_param[v];
                for (int c = 0; c < 3; c++) {
                    if (c > 0) fprintf(ortho_file, ",");
                    // Format: re.a.num/re.a.den+re.b.num/re.b.den*sqrt(re.rad)+i(im.a.num/im.a.den+im.b.num/im.b.den*sqrt(im.rad))
                    fprintf(ortho_file, "%ld/%ld+%ld/%ld*sqrt(%d)+i(%ld/%ld+%ld/%ld*sqrt(%d))",
                           vec[c].re.a.num, vec[c].re.a.den,
                           vec[c].re.b.num, vec[c].re.b.den, vec[c].re.rad,
                           vec[c].im.a.num, vec[c].im.a.den,
                           vec[c].im.b.num, vec[c].im.b.den, vec[c].im.rad);
                }
            } else {
                // Write integer vectors
                const auto& vec = vectors_int[v];
                for (size_t c = 0; c < vec.size(); c++) {
                    if (c > 0) fprintf(ortho_file, ",");
                    fprintf(ortho_file, "%d", vec[c]);
                }
            }
            fprintf(ortho_file, "]");
        }

        fprintf(ortho_file, "\n");
        fflush(ortho_file);

        if (print_statement) {
            DEBUG_PRINT("Wrote orthogonality witness to file for " << vertices_set.size() << " vertices");
        }
    }
}