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
    string file_path;

    StreamState(int idx, const string& path) : stream_idx(idx), head(0), tail(0), eof(false), f(nullptr), file_path(path) {
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

    void rewind() {
        if (f) {
            fseek(f, 0, SEEK_SET);
            head = 0;
            tail = 0;
            eof = false;
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
    // PPMI & STATS CALCULATION MODE (3-Pass Streaming Merger)
    // ────────────────────────────────────────────────────────
    
    // ── PASS 1: Merge streams to build count histogram and find max token ID ──
    vector<uint64_t> global_histogram(1000000, 0);
    unordered_map<uint32_t, uint64_t> large_counts_map;

    uint32_t current_key = 0xFFFFFFFF;
    uint64_t current_count = 0;
    uint64_t total_pairs = 0;
    uint32_t max_token_id = 0;

    while (!pq.empty()) {
        MergeItem top = pq.top();
        pq.pop();

        if (top.key == current_key) {
            current_count += top.count;
        } else {
            if (current_key != 0xFFFFFFFF) {
                uint32_t final_c = (current_count > 0xFFFFFFFF) ? 0xFFFFFFFF : (uint32_t)current_count;
                total_pairs++;
                
                uint16_t u = (uint16_t)(current_key >> 16);
                uint16_t v = (uint16_t)(current_key & 0xFFFF);
                if (u > max_token_id) max_token_id = u;
                if (v > max_token_id) max_token_id = v;

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
        total_pairs++;
        
        uint16_t u = (uint16_t)(current_key >> 16);
        uint16_t v = (uint16_t)(current_key & 0xFFFF);
        if (u > max_token_id) max_token_id = u;
        if (v > max_token_id) max_token_id = v;

        if (final_c < 1000000) {
            global_histogram[final_c]++;
        } else {
            large_counts_map[final_c]++;
        }
    }

    uint32_t vocab_size = max_token_id + 1;
    cerr << "STATS:TOTAL_PAIRS=" << total_pairs << ";DYNAMIC_VOCAB_SIZE=" << vocab_size << endl;

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

    // Free histogram memory
    global_histogram.clear();
    global_histogram.shrink_to_fit();
    large_counts_map.clear();

    // ── PASS 2: Rewind streams, merge again, filter, calculate marginals, and buffer to temp disk ──
    for (auto s : streams) {
        s->rewind();
    }
    pq = priority_queue<MergeItem, vector<MergeItem>, greater<MergeItem>>();
    for (int s = 0; s < num_streams; ++s) {
        Record r;
        if (streams[s]->next(r)) {
            pq.push({r.key(), r.count, s});
        }
    }

    // Open temporary file for storing filtered records
    string temp_file_path = string(output_path) + ".tmp";
    FILE* f_temp = fopen(temp_file_path.c_str(), "wb");
    if (!f_temp) {
        cerr << "ERROR: cannot open temporary storage " << temp_file_path << endl;
        return 1;
    }
    setvbuf(f_temp, NULL, _IOFBF, 16 * 1024 * 1024);

    vector<Record> write_buf;
    write_buf.reserve(65536);

    vector<uint64_t> marginal_counts(vocab_size, 0);
    uint64_t ppmi_total_sum = 0;
    uint64_t remaining_pairs = 0;

    current_key = 0xFFFFFFFF;
    current_count = 0;

    while (!pq.empty()) {
        MergeItem top = pq.top();
        pq.pop();

        if (top.key == current_key) {
            current_count += top.count;
        } else {
            if (current_key != 0xFFFFFFFF) {
                uint32_t final_c = (current_count > 0xFFFFFFFF) ? 0xFFFFFFFF : (uint32_t)current_count;
                if (final_c >= threshold) {
                    uint16_t u = (uint16_t)(current_key >> 16);
                    uint16_t v = (uint16_t)(current_key & 0xFFFF);
                    
                    marginal_counts[u] += final_c;
                    marginal_counts[v] += final_c;
                    ppmi_total_sum += 2 * final_c;
                    remaining_pairs++;

                    write_buf.push_back({u, v, final_c});
                    if (write_buf.size() >= 65536) {
                        fwrite(write_buf.data(), sizeof(Record), write_buf.size(), f_temp);
                        write_buf.clear();
                    }
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
        if (final_c >= threshold) {
            uint16_t u = (uint16_t)(current_key >> 16);
            uint16_t v = (uint16_t)(current_key & 0xFFFF);
            
            marginal_counts[u] += final_c;
            marginal_counts[v] += final_c;
            ppmi_total_sum += 2 * final_c;
            remaining_pairs++;

            write_buf.push_back({u, v, final_c});
        }
    }
    if (!write_buf.empty()) {
        fwrite(write_buf.data(), sizeof(Record), write_buf.size(), f_temp);
    }
    fclose(f_temp);

    // Clean up readers
    for (auto s : streams) {
        delete s;
    }
    streams.clear();

    cerr << "STATS:REMAINING_PAIRS=" << remaining_pairs << endl;

    // ── PASS 3: Read filtered binary records sequentially from temp disk and build heaps ──
    vector<HeapItem> flat_heaps(vocab_size * 60);
    vector<int> heap_sizes(vocab_size, 0);

    double log_total_sum = log((double)ppmi_total_sum);

    FILE* f_temp_in = fopen(temp_file_path.c_str(), "rb");
    if (!f_temp_in) {
        cerr << "ERROR: cannot open temporary storage for reading " << temp_file_path << endl;
        return 1;
    }
    setvbuf(f_temp_in, NULL, _IOFBF, 16 * 1024 * 1024);

    vector<Record> read_buf(65536);
    while (true) {
        size_t n = fread(read_buf.data(), sizeof(Record), read_buf.size(), f_temp_in);
        if (n <= 0) break;

        for (size_t k = 0; k < n; ++k) {
            const auto& r = read_buf[k];
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
    }
    fclose(f_temp_in);
    remove(temp_file_path.c_str()); // Clean up temp file from disk

    // ── Write final PPMI CSV file ──
    FILE* f_out = fopen(output_path, "w");
    if (!f_out) {
        cerr << "ERROR: cannot open output file " << output_path << endl;
        return 1;
    }

    // Set 16 MB sequential buffer for fast text output
    setvbuf(f_out, NULL, _IOFBF, 16 * 1024 * 1024);

    fprintf(f_out, "i,j,ppmi\n");

    for (uint32_t i = 0; i < vocab_size; ++i) {
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
