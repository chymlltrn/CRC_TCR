// Stream MatrixMarket triplets into patient/subtype pseudobulks without
// materializing the cell-by-gene matrix. Gene and cell maps are zero-based.
#include <zlib.h>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Reader {
    gzFile f;
    std::vector<unsigned char> buf;
    size_t pos = 0, size = 0;
    explicit Reader(const char* path): f(gzopen(path, "rb")), buf(16*1024*1024) {
        if (!f) throw std::runtime_error("Cannot open gzip input");
        gzbuffer(f, 16*1024*1024);
    }
    ~Reader() { gzclose(f); }
    inline int get() {
        if (pos == size) {
            int got = gzread(f, buf.data(), buf.size());
            if (got < 0) throw std::runtime_error("gzip read error");
            if (got == 0) return -1;
            pos = 0; size = got;
        }
        return buf[pos++];
    }
    std::string line() {
        std::string s; int c;
        while ((c=get()) >= 0 && c != '\n') s.push_back(c);
        return s;
    }
    inline int64_t number() {
        int c;
        do { c=get(); } while (c >= 0 && c <= 32);
        if (c < 0) return -1;
        if (c < '0' || c > '9') throw std::runtime_error("Non-positive-integer matrix token");
        int64_t out=0;
        do { out=out*10+c-'0'; c=get(); } while (c>='0' && c<='9');
        if (c > 32) throw std::runtime_error("Malformed integer token");
        return out;
    }
};

template<class T> std::vector<T> read_binary(const std::string& path, size_t n) {
    std::vector<T> data(n); std::ifstream f(path, std::ios::binary);
    f.read(reinterpret_cast<char*>(data.data()), n*sizeof(T));
    if (!f || f.peek()!=EOF) throw std::runtime_error("Wrong map size: "+path);
    return data;
}
template<class T> void write_binary(const std::string& path, const std::vector<T>& data) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size()*sizeof(T));
    if (!f) throw std::runtime_error("Cannot write: "+path);
}

int main(int argc, char** argv) {
    try {
        if (argc!=7) throw std::runtime_error("args: raw.gz gene_map.bin cell_group.bin genes groups output_prefix");
        int genes=std::stoi(argv[4]), groups=std::stoi(argv[5]);
        Reader r(argv[1]); std::string header=r.line();
        if (header!="%%MatrixMarket matrix coordinate integer general") throw std::runtime_error("Unexpected matrix format");
        do { header=r.line(); } while (!header.empty() && header[0]=='%');
        int rows, cells; int64_t expected;
        std::istringstream(header)>>rows>>cells>>expected;
        auto gene_map=read_binary<int32_t>(argv[2], rows);
        auto cell_group=read_binary<int32_t>(argv[3], cells);
        std::vector<int64_t> pb(size_t(genes)*groups,0), totals(cells,0);
        int64_t all_total=0, selected_total=0, selected_nnz=0;
        for (int64_t i=0; i<expected; ++i) {
            int64_t row=r.number()-1, cell=r.number()-1, count=r.number();
            if (row<0 || row>=rows || cell<0 || cell>=cells || count<1) throw std::runtime_error("Invalid matrix entry");
            totals[cell]+=count; all_total+=count;
            int group=cell_group[cell], gene=gene_map[row];
            if (group>=0 && gene>=0) {
                if (group>=groups || gene>=genes) throw std::runtime_error("Out-of-bounds map");
                pb[size_t(group)*genes+gene]+=count;
                selected_total+=count; selected_nnz++;
            }
            if ((i+1)%100000000==0) std::cerr<<"triplets "<<i+1<<" / "<<expected<<std::endl;
        }
        if (r.number()!=-1) throw std::runtime_error("Unexpected extra matrix entries");
        std::string prefix=argv[6];
        write_binary(prefix+"_counts.bin", pb);
        write_binary(prefix+"_cell_totals.bin", totals);
        std::ofstream j(prefix+"_stream.json");
        j<<"{\"raw_rows\":"<<rows<<",\"raw_cells\":"<<cells<<",\"raw_nnz\":"<<expected
         <<",\"raw_total_counts\":"<<all_total<<",\"selected_total_counts\":"<<selected_total
         <<",\"selected_nnz\":"<<selected_nnz<<",\"genes\":"<<genes<<",\"groups\":"<<groups<<"}\n";
    } catch(const std::exception& e) { std::cerr<<e.what()<<std::endl; return 1; }
}
