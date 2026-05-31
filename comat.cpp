#include <iostream>
#include <vector>
#include <queue>
#include <cstdint>
#include <algorithm>
#include <unordered_map>
#include <string>
#include <cmath>
#include <fstream>
#include <cstring>
#include <cstdio>

using namespace std;

#pragma pack(push, 1)
struct Record {
    uint16_t i;
    uint16_t j;
    uint32_t count;
    uint32_t key() const {
        return ((uint32_t)i << 16) | j;
    }
};
#pragma pack(pop)

struct StreamState {
    int stream_idx;
    vector<Record> buffer;
    int head;
    int tail;
    bool eof;
    FILE* f;

    StreamState(int idx, const string& path) : stream_idx(idx), head(0), tail(0), eof(false), f(nullptr) {
        // Caching 2M records = 16 MB buffer per stream to leverage large RAM on GitHub runner.
        // For 88 streams, this will consume ~1.4 GB RAM, which is extremely safe on a 16 GB machine.
        buffer.resize(2097152); 
        f = fopen(path.c_str(), "rb");
        if (!f) {
            cerr << "ERROR: Failed to open stream file: " << path << endl;
            eof = true;
        } else {
            // Set 16 MB sequential buffer to optimize disk reads
            setvbuf(f, NULL, _IOFBF, 16 * 1024 * 1024);
        }
    }

    ~StreamState() {
        if (f) {
            fclose(f);
        }
    }

    bool next(Record& r) {
        if (head >= tail) {
            if (eof) return false;
            size_t n = fread(buffer.data(), sizeof(Record), buffer.size(), f);
            if (n <= 0) {
                eof = true;
                return false;
            }
            tail = (int)n;
            head = 0;
        }
        r = buffer[head++];
        return true;
    }
};

struct MergeItem {
    uint32_t key;
    uint32_t count;
    int stream_idx;

    bool operator>(const MergeItem& other) const {
        return key > other.key;
    }
};

struct HeapItem {
    float value; // PPMI value
    uint16_t neighbor;

    bool operator>(const HeapItem& other) const {
        if (value == other.value) {
            return neighbor > other.neighbor;
        }
        return value > other.value;
    }
};

// Global states in RAM (pre-allocated)
vector<Record> global_records;

int main(int argc, char* argv[]) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " <file_list_path> <mode: intermediate|ppmi> <percentile> <output_path>" << endl;
        return 1;
    }

    const char* file_list_path = argv[1];
    string mode = argv[2];
    double percentile = stod(argv[3]);
    const char* output_path = argv[4];

    // Read list of local binary files
    ifstream flist(file_list_path);
    if (!flist.is_open()) {
        cerr << "ERROR: Cannot open file list: " << file_list_path << endl;
        return 1;
    }

    vector<string> file_paths;
    string line;
    while (getline(flist, line)) {
        if (!line.empty()) {
            file_paths.push_back(line);
        }
    }
    flist.close();

    int num_streams = (int)file_paths.size();
    if (num_streams == 0) {
        cerr << "ERROR: No input files specified in file list." << endl;
        return 1;
    }

    cout << "Merging " << num_streams << " local streams in C++ (mode: " << mode << ")..." << endl;

    // Initialize stream readers
    vector<StreamState*> streams;
    priority_queue<MergeItem, vector<MergeItem>, greater<MergeItem>> pq;

    for (int s = 0; s < num_streams; ++s) {
        StreamState* state = new StreamState(s, file_paths[s]);
        Record r;
        if (state->next(r)) {
            pq.push({r.key(), r.count, s});
        }
        streams.push_back(state);
    }

    // ────────────────────────────────────────────────────────
    // INTERMEDIATE MERGE MODE
    // ────────────────────────────────────────────────────────
    if (mode == "intermediate") {
        FILE* f_out = fopen(output_path, "wb");
        if (!f_out) {
            cerr << "ERROR: cannot open output file " << output_path << endl;
            return 1;
        }
        setvbuf(f_out, NULL, _IOFBF, 16 * 1024 * 1024);

        vector<Record> write_buf;
        write_buf.reserve(65536);

        uint32_t current_key = 0xFFFFFFFF;
        uint64_t current_count = 0;

        while (!pq.empty()) {
            MergeItem top = pq.top();
            pq.pop();

            if (top.key == current_key) {
                current_count += top.count;
            } else {
                if (current_key != 0xFFFFFFFF) {
                    uint32_t final_c = (current_count > 0xFFFFFFFF) ? 0xFFFFFFFF : (uint32_t)current_count;
                    write_buf.push_back({(uint16_t)(current_key >> 16), (uint16_t)(current_key & 0xFFFF), final_c});
                    if (write_buf.size() >= 65536) {
                        fwrite(write_buf.data(), sizeof(Record), write_buf.size(), f_out);
                        write_buf.clear();
                    }
                }
                current_key = top.key;
                current_count = top.count;
            }

            int s = top.stream_idx;
            Record r;
            if (streams[s]->next(r)) {
                pq.push({r.key(), r.count, s});
            }
        }

        if (current_key != 0xFFFFFFFF) {
            uint32_t final_c = (current_count > 0xFFFFFFFF) ? 0xFFFFFFFF : (uint32_t)current_count;
            write_buf.push_back({(uint16_t)(current_key >> 16), (uint16_t)(current_key & 0xFFFF), final_c});
        }
        if (!write_buf.empty()) {
            fwrite(write_buf.data(), sizeof(Record), write_buf.size(), f_out);
        }

        fclose(f_out);

        for (auto s : streams) {
            delete s;
        }

        cout << "Intermediate merge complete. Saved to " << output_path << endl;
        return 0;
    }

    // ────────────────────────────────────────────────────────
    // PPMI & STATS CALCULATION MODE
    // ────────────────────────────────────────────────────────
    // Reserve memory up-front (200 million records = 1.6 GB RAM) to eliminate reallocation / resize overhead
    global_records.reserve(200000000);

    uint32_t current_key = 0xFFFFFFFF;
    uint64_t current_count = 0;

    // Histogram arrays for O(1) memory statistics calculation
    vector<uint64_t> global_histogram(1000000, 0);
    unordered_map<uint32_t, uint64_t> large_counts_map;

    while (!pq.empty()) {
        MergeItem top = pq.top();
        pq.pop();

        if (top.key == current_key) {
            current_count += top.count;
        } else {
            if (current_key != 0xFFFFFFFF) {
                uint32_t final_c = (current_count > 0xFFFFFFFF) ? 0xFFFFFFFF : (uint32_t)current_count;
                global_records.push_back({(uint16_t)(current_key >> 16), (uint16_t)(current_key & 0xFFFF), final_c});
                
                if (final_c < 1000000) {
                    global_histogram[final_c]++;
                } else {
                    large_counts_map[final_c]++;
                }
            }
            current_key = top.key;
            current_count = top.count;
        }

        int s = top.stream_idx;
        Record r;
        if (streams[s]->next(r)) {
            pq.push({r.key(), r.count, s});
        }
    }

    if (current_key != 0xFFFFFFFF) {
        uint32_t final_c = (current_count > 0xFFFFFFFF) ? 0xFFFFFFFF : (uint32_t)current_count;
        global_records.push_back({(uint16_t)(current_key >> 16), (uint16_t)(current_key & 0xFFFF), final_c});
        
        if (final_c < 1000000) {
            global_histogram[final_c]++;
        } else {
            large_counts_map[final_c]++;
        }
    }

    // Clean up streams
    for (auto s : streams) {
        delete s;
    }

    uint64_t total_pairs = global_records.size();
    uint64_t sum_counts = 0;

    for (uint32_t c = 1; c < 1000000; ++c) {
        sum_counts += global_histogram[c] * c;
    }
    for (const auto& p : large_counts_map) {
        sum_counts += p.second * p.first;
    }

    double mean_val = total_pairs > 0 ? (double)sum_counts / total_pairs : 0.0;
    
    // Mode
    uint64_t max_freq = 0;
    int mode_val = 0;
    for (uint32_t c = 1; c < 1000000; ++c) {
        if (global_histogram[c] > max_freq) {
            max_freq = global_histogram[c];
            mode_val = c;
        }
    }
    for (const auto& p : large_counts_map) {
        if (p.second > max_freq) {
            max_freq = p.second;
            mode_val = p.first;
        }
    }

    // Median
    uint64_t target = total_pairs / 2;
    uint64_t cumulative = 0;
    int median_val = 0;

    for (uint32_t c = 1; c < 1000000; ++c) {
        cumulative += global_histogram[c];
        if (cumulative >= target) {
            median_val = c;
            break;
        }
    }
    if (median_val == 0) {
        vector<pair<uint32_t, uint64_t>> large_pairs(large_counts_map.begin(), large_counts_map.end());
        sort(large_pairs.begin(), large_pairs.end());
        for (const auto& p : large_pairs) {
            cumulative += p.second;
            if (cumulative >= target) {
                median_val = p.first;
                break;
            }
        }
    }

    cerr << "STATS:MEAN=" << mean_val << ";MEDIAN=" << median_val << ";MODE=" << mode_val << endl;

    // Apply percentile thresholding
    uint64_t target_thresh = (uint64_t)(total_pairs * percentile);
    uint64_t cumulative_thresh = 0;
    uint32_t threshold = 0;

    for (uint32_t c = 1; c < 1000000; ++c) {
        cumulative_thresh += global_histogram[c];
        if (cumulative_thresh >= target_thresh) {
            threshold = c;
            break;
        }
    }
    if (threshold == 0) {
        vector<pair<uint32_t, uint64_t>> large_pairs(large_counts_map.begin(), large_counts_map.end());
        sort(large_pairs.begin(), large_pairs.end());
        for (const auto& p : large_pairs) {
            cumulative_thresh += p.second;
            if (cumulative_thresh >= target_thresh) {
                threshold = p.first;
                break;
            }
        }
    }

    cerr << "STATS:THRESHOLD=" << threshold << endl;

    // Filter in-place
    auto it = remove_if(global_records.begin(), global_records.end(), [threshold](const Record& r) {
        return r.count < threshold;
    });
    global_records.erase(it, global_records.end());
    global_records.shrink_to_fit();

    cerr << "STATS:REMAINING_PAIRS=" << global_records.size() << endl;

    // Calculate marginals
    vector<uint64_t> marginal_counts(52022, 0);
    uint64_t ppmi_total_sum = 0;

    for (const auto& r : global_records) {
        marginal_counts[r.i] += r.count;
        marginal_counts[r.j] += r.count;
        ppmi_total_sum += 2 * r.count;
    }

    // Build flat cache-friendly representation for PPMI heaps (52022 tokens, max 60 neighbors each)
    // Allocates ~25 MB as a single block in RAM, avoiding 52,022 separate mallocs.
    vector<HeapItem> flat_heaps(52022 * 60);
    vector<int> heap_sizes(52022, 0);

    double log_total_sum = log((double)ppmi_total_sum);

    for (const auto& r : global_records) {
        uint16_t u = r.i;
        uint16_t v = r.j;
        uint32_t count = r.count;

        uint64_t mu = marginal_counts[u];
        uint64_t mv = marginal_counts[v];

        if (mu == 0 || mv == 0 || ppmi_total_sum == 0) continue;

        double ppmi_val = log((double)count) + log_total_sum - log((double)mu) - log((double)mv);
        if (ppmi_val <= 0.0) continue;

        float f_ppmi = (float)ppmi_val;

        // Heap operations on flat_heaps for u
        int& su = heap_sizes[u];
        HeapItem* hu = &flat_heaps[u * 60];
        if (su < 60) {
            hu[su] = {f_ppmi, v};
            su++;
            push_heap(hu, hu + su, greater<HeapItem>());
        } else if (f_ppmi > hu[0].value) {
            pop_heap(hu, hu + su, greater<HeapItem>());
            hu[59] = {f_ppmi, v};
            push_heap(hu, hu + su, greater<HeapItem>());
        }

        // Heap operations on flat_heaps for v
        int& sv = heap_sizes[v];
        HeapItem* hv = &flat_heaps[v * 60];
        if (sv < 60) {
            hv[sv] = {f_ppmi, u};
            sv++;
            push_heap(hv, hv + sv, greater<HeapItem>());
        } else if (f_ppmi > hv[0].value) {
            pop_heap(hv, hv + sv, greater<HeapItem>());
            hv[59] = {f_ppmi, u};
            push_heap(hv, hv + sv, greater<HeapItem>());
        }
    }

    // Write final PPMI CSV file
    FILE* f_out = fopen(output_path, "w");
    if (!f_out) {
        cerr << "ERROR: cannot open output file " << output_path << endl;
        return 1;
    }

    // Set 16 MB sequential buffer for fast text output
    setvbuf(f_out, NULL, _IOFBF, 16 * 1024 * 1024);

    fprintf(f_out, "i,j,ppmi\n");

    for (uint16_t i = 0; i < 52022; ++i) {
        int su = heap_sizes[i];
        if (su == 0) continue;

        HeapItem* hu = &flat_heaps[i * 60];
        sort(hu, hu + su, [](const HeapItem& a, const HeapItem& b) {
            if (a.value == b.value) {
                return a.neighbor < b.neighbor;
            }
            return a.value > b.value;
        });

        for (int k = 0; k < su; ++k) {
            fprintf(f_out, "%d,%d,%f\n", i, hu[k].neighbor, hu[k].value);
        }
    }

    fclose(f_out);

    cerr << "PPMI Table Trim and save complete." << endl;
    return 0;
}
