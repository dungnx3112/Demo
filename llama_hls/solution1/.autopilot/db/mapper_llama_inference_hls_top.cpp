#include "hls_signal_handler.h"
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <vector>
#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_directio.h"
#include "hls_stream.h"
using namespace std;

namespace hls::sim
{
  template<size_t n>
  struct Byte {
    unsigned char a[n];

    Byte()
    {
      for (size_t i = 0; i < n; ++i) {
        a[i] = 0;
      }
    }

    template<typename T>
    Byte<n>& operator= (const T &val)
    {
      std::memcpy(a, &val, n);
      return *this;
    }
  };

  struct SimException : public std::exception {
    const std::string msg;
    const size_t line;
    SimException(const std::string &msg, const size_t line)
      : msg(msg), line(line)
    {
    }
  };

  void errExit(const size_t line, const std::string &msg)
  {
    std::string s;
    s += "ERROR";
//  s += '(';
//  s += __FILE__;
//  s += ":";
//  s += std::to_string(line);
//  s += ')';
    s += ": ";
    s += msg;
    s += "\n";
    fputs(s.c_str(), stderr);
    exit(1);
  }
}


namespace hls::sim
{
  struct Buffer {
    char *first;
    Buffer(char *addr) : first(addr)
    {
    }
  };

  struct DBuffer : public Buffer {
    static const size_t total = 1<<10;
    size_t ufree;

    DBuffer(size_t usize) : Buffer(nullptr), ufree(total)
    {
      first = new char[usize*ufree];
    }

    ~DBuffer()
    {
      delete[] first;
    }
  };

  struct CStream {
    char *front;
    char *back;
    size_t num;
    size_t usize;
    std::list<Buffer*> bufs;
    bool dynamic;

    CStream() : front(nullptr), back(nullptr),
                num(0), usize(0), dynamic(true)
    {
    }

    ~CStream()
    {
      for (Buffer *p : bufs) {
        delete p;
      }
    }

    template<typename T>
    T* data()
    {
      return (T*)front;
    }

    template<typename T>
    void transfer(hls::stream<T> *param)
    {
      while (!empty()) {
        param->write(*(T*)nextRead());
      }
    }

    bool empty();
    char* nextRead();
    char* nextWrite();
  };

  bool CStream::empty()
  {
    return num == 0;
  }

  char* CStream::nextRead()
  {
    assert(num > 0);
    char *res = front;
    front += usize;
    if (dynamic) {
      if (++static_cast<DBuffer*>(bufs.front())->ufree == DBuffer::total) {
        if (bufs.size() > 1) {
          bufs.pop_front();
          front = bufs.front()->first;
        } else {
          front = back = bufs.front()->first;
        }
      }
    }
    --num;
    return res;
  }

  char* CStream::nextWrite()
  {
    if (dynamic) {
      if (static_cast<DBuffer*>(bufs.back())->ufree == 0) {
        bufs.push_back(new DBuffer(usize));
        back = bufs.back()->first;
      }
      --static_cast<DBuffer*>(bufs.back())->ufree;
    }
    char *res = back;
    back += usize;
    ++num;
    return res;
  }

  std::list<CStream> streams;
  std::map<char*, CStream*> prebuilt;

  CStream* createStream(size_t usize)
  {
    streams.emplace_front();
    CStream &s = streams.front();
    {
      s.dynamic = true;
      s.bufs.push_back(new DBuffer(usize));
      s.front = s.bufs.back()->first;
      s.back = s.front;
      s.num = 0;
      s.usize = usize;
    }
    return &s;
  }

  template<typename T>
  CStream* createStream(hls::stream<T> *param)
  {
    CStream *s = createStream(sizeof(T));
    {
      s->dynamic = true;
      while (!param->empty()) {
        T data = param->read();
        memcpy(s->nextWrite(), (char*)&data, sizeof(T));
      }
      prebuilt[s->front] = s;
    }
    return s;
  }

  template<typename T>
  CStream* createStream(T *param, size_t usize)
  {
    streams.emplace_front();
    CStream &s = streams.front();
    {
      s.dynamic = false;
      s.bufs.push_back(new Buffer((char*)param));
      s.front = s.back = s.bufs.back()->first;
      s.usize = usize;
      s.num = ~0UL;
    }
    prebuilt[s.front] = &s;
    return &s;
  }

  CStream* findStream(char *buf)
  {
    return prebuilt.at(buf);
  }
}
class AESL_RUNTIME_BC {
  public:
    AESL_RUNTIME_BC(const char* name) {
      file_token.open( name);
      if (!file_token.good()) {
        cout << "Failed to open tv file " << name << endl;
        exit (1);
      }
      file_token >> mName;//[[[runtime]]]
    }
    ~AESL_RUNTIME_BC() {
      file_token.close();
    }
    int read_size () {
      int size = 0;
      file_token >> mName;//[[transaction]]
      file_token >> mName;//transaction number
      file_token >> mName;//pop_size
      size = atoi(mName.c_str());
      file_token >> mName;//[[/transaction]]
      return size;
    }
  public:
    fstream file_token;
    string mName;
};
using hls::sim::Byte;
extern "C" void llama_inference_hls_top(Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, Byte<4>*, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int, int);
extern "C" void apatb_llama_inference_hls_top_hw(volatile void * __xlx_apatb_param_input_embedding, volatile void * __xlx_apatb_param_output_logits, volatile void * __xlx_apatb_param_weight_token_embedding, volatile void * __xlx_apatb_param_weight_attention_wq, volatile void * __xlx_apatb_param_weight_attention_wk, volatile void * __xlx_apatb_param_weight_attention_wv, volatile void * __xlx_apatb_param_weight_attention_wo, volatile void * __xlx_apatb_param_weight_ffn_w1, volatile void * __xlx_apatb_param_weight_ffn_w2, volatile void * __xlx_apatb_param_weight_ffn_w3, volatile void * __xlx_apatb_param_weight_attention_norm, volatile void * __xlx_apatb_param_weight_ffn_norm, volatile void * __xlx_apatb_param_weight_final_norm, volatile void * __xlx_apatb_param_k_cache, volatile void * __xlx_apatb_param_v_cache, volatile void * __xlx_apatb_param_cos_table, volatile void * __xlx_apatb_param_sin_table, int __xlx_apatb_param_position, int __xlx_apatb_param_max_position) {
using hls::sim::createStream;
  // Collect __xlx_input_embedding__tmp_vec
std::vector<Byte<4>> __xlx_input_embedding__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_input_embedding__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_input_embedding)[i]);
}
  int __xlx_size_param_input_embedding = 1;
  int __xlx_offset_param_input_embedding = 0;
  int __xlx_offset_byte_param_input_embedding = 0*4;
  // Collect __xlx_output_logits__tmp_vec
std::vector<Byte<4>> __xlx_output_logits__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_output_logits__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_output_logits)[i]);
}
  int __xlx_size_param_output_logits = 1;
  int __xlx_offset_param_output_logits = 0;
  int __xlx_offset_byte_param_output_logits = 0*4;
  // Collect __xlx_weight_token_embedding__tmp_vec
std::vector<Byte<4>> __xlx_weight_token_embedding__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_weight_token_embedding__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_weight_token_embedding)[i]);
}
  int __xlx_size_param_weight_token_embedding = 1;
  int __xlx_offset_param_weight_token_embedding = 0;
  int __xlx_offset_byte_param_weight_token_embedding = 0*4;
  // Collect __xlx_weight_attention_wq__tmp_vec
std::vector<Byte<4>> __xlx_weight_attention_wq__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_weight_attention_wq__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_weight_attention_wq)[i]);
}
  int __xlx_size_param_weight_attention_wq = 1;
  int __xlx_offset_param_weight_attention_wq = 0;
  int __xlx_offset_byte_param_weight_attention_wq = 0*4;
  // Collect __xlx_weight_attention_wk__tmp_vec
std::vector<Byte<4>> __xlx_weight_attention_wk__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_weight_attention_wk__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_weight_attention_wk)[i]);
}
  int __xlx_size_param_weight_attention_wk = 1;
  int __xlx_offset_param_weight_attention_wk = 0;
  int __xlx_offset_byte_param_weight_attention_wk = 0*4;
  // Collect __xlx_weight_attention_wv__tmp_vec
std::vector<Byte<4>> __xlx_weight_attention_wv__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_weight_attention_wv__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_weight_attention_wv)[i]);
}
  int __xlx_size_param_weight_attention_wv = 1;
  int __xlx_offset_param_weight_attention_wv = 0;
  int __xlx_offset_byte_param_weight_attention_wv = 0*4;
  // Collect __xlx_weight_attention_wo__tmp_vec
std::vector<Byte<4>> __xlx_weight_attention_wo__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_weight_attention_wo__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_weight_attention_wo)[i]);
}
  int __xlx_size_param_weight_attention_wo = 1;
  int __xlx_offset_param_weight_attention_wo = 0;
  int __xlx_offset_byte_param_weight_attention_wo = 0*4;
  // Collect __xlx_weight_ffn_w1__tmp_vec
std::vector<Byte<4>> __xlx_weight_ffn_w1__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_weight_ffn_w1__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_weight_ffn_w1)[i]);
}
  int __xlx_size_param_weight_ffn_w1 = 1;
  int __xlx_offset_param_weight_ffn_w1 = 0;
  int __xlx_offset_byte_param_weight_ffn_w1 = 0*4;
  // Collect __xlx_weight_ffn_w2__tmp_vec
std::vector<Byte<4>> __xlx_weight_ffn_w2__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_weight_ffn_w2__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_weight_ffn_w2)[i]);
}
  int __xlx_size_param_weight_ffn_w2 = 1;
  int __xlx_offset_param_weight_ffn_w2 = 0;
  int __xlx_offset_byte_param_weight_ffn_w2 = 0*4;
  // Collect __xlx_weight_ffn_w3__tmp_vec
std::vector<Byte<4>> __xlx_weight_ffn_w3__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_weight_ffn_w3__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_weight_ffn_w3)[i]);
}
  int __xlx_size_param_weight_ffn_w3 = 1;
  int __xlx_offset_param_weight_ffn_w3 = 0;
  int __xlx_offset_byte_param_weight_ffn_w3 = 0*4;
  // Collect __xlx_weight_attention_norm__tmp_vec
std::vector<Byte<4>> __xlx_weight_attention_norm__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_weight_attention_norm__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_weight_attention_norm)[i]);
}
  int __xlx_size_param_weight_attention_norm = 1;
  int __xlx_offset_param_weight_attention_norm = 0;
  int __xlx_offset_byte_param_weight_attention_norm = 0*4;
  // Collect __xlx_weight_ffn_norm__tmp_vec
std::vector<Byte<4>> __xlx_weight_ffn_norm__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_weight_ffn_norm__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_weight_ffn_norm)[i]);
}
  int __xlx_size_param_weight_ffn_norm = 1;
  int __xlx_offset_param_weight_ffn_norm = 0;
  int __xlx_offset_byte_param_weight_ffn_norm = 0*4;
  // Collect __xlx_weight_final_norm__tmp_vec
std::vector<Byte<4>> __xlx_weight_final_norm__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_weight_final_norm__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_weight_final_norm)[i]);
}
  int __xlx_size_param_weight_final_norm = 1;
  int __xlx_offset_param_weight_final_norm = 0;
  int __xlx_offset_byte_param_weight_final_norm = 0*4;
  // Collect __xlx_k_cache__tmp_vec
std::vector<Byte<4>> __xlx_k_cache__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_k_cache__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_k_cache)[i]);
}
  int __xlx_size_param_k_cache = 1;
  int __xlx_offset_param_k_cache = 0;
  int __xlx_offset_byte_param_k_cache = 0*4;
  // Collect __xlx_v_cache__tmp_vec
std::vector<Byte<4>> __xlx_v_cache__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_v_cache__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_v_cache)[i]);
}
  int __xlx_size_param_v_cache = 1;
  int __xlx_offset_param_v_cache = 0;
  int __xlx_offset_byte_param_v_cache = 0*4;
  // Collect __xlx_cos_table__tmp_vec
std::vector<Byte<4>> __xlx_cos_table__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_cos_table__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_cos_table)[i]);
}
  int __xlx_size_param_cos_table = 1;
  int __xlx_offset_param_cos_table = 0;
  int __xlx_offset_byte_param_cos_table = 0*4;
  // Collect __xlx_sin_table__tmp_vec
std::vector<Byte<4>> __xlx_sin_table__tmp_vec;
for (size_t i = 0; i < 1; ++i){
__xlx_sin_table__tmp_vec.push_back(((Byte<4>*)__xlx_apatb_param_sin_table)[i]);
}
  int __xlx_size_param_sin_table = 1;
  int __xlx_offset_param_sin_table = 0;
  int __xlx_offset_byte_param_sin_table = 0*4;
  // DUT call
  llama_inference_hls_top(__xlx_input_embedding__tmp_vec.data(), __xlx_output_logits__tmp_vec.data(), __xlx_weight_token_embedding__tmp_vec.data(), __xlx_weight_attention_wq__tmp_vec.data(), __xlx_weight_attention_wk__tmp_vec.data(), __xlx_weight_attention_wv__tmp_vec.data(), __xlx_weight_attention_wo__tmp_vec.data(), __xlx_weight_ffn_w1__tmp_vec.data(), __xlx_weight_ffn_w2__tmp_vec.data(), __xlx_weight_ffn_w3__tmp_vec.data(), __xlx_weight_attention_norm__tmp_vec.data(), __xlx_weight_ffn_norm__tmp_vec.data(), __xlx_weight_final_norm__tmp_vec.data(), __xlx_k_cache__tmp_vec.data(), __xlx_v_cache__tmp_vec.data(), __xlx_cos_table__tmp_vec.data(), __xlx_sin_table__tmp_vec.data(), __xlx_offset_byte_param_input_embedding, __xlx_offset_byte_param_output_logits, __xlx_offset_byte_param_weight_token_embedding, __xlx_offset_byte_param_weight_attention_wq, __xlx_offset_byte_param_weight_attention_wk, __xlx_offset_byte_param_weight_attention_wv, __xlx_offset_byte_param_weight_attention_wo, __xlx_offset_byte_param_weight_ffn_w1, __xlx_offset_byte_param_weight_ffn_w2, __xlx_offset_byte_param_weight_ffn_w3, __xlx_offset_byte_param_weight_attention_norm, __xlx_offset_byte_param_weight_ffn_norm, __xlx_offset_byte_param_weight_final_norm, __xlx_offset_byte_param_k_cache, __xlx_offset_byte_param_v_cache, __xlx_offset_byte_param_cos_table, __xlx_offset_byte_param_sin_table, __xlx_apatb_param_position, __xlx_apatb_param_max_position);
// print __xlx_apatb_param_input_embedding
for (size_t i = 0; i < __xlx_size_param_input_embedding; ++i) {
((Byte<4>*)__xlx_apatb_param_input_embedding)[i] = __xlx_input_embedding__tmp_vec[__xlx_offset_param_input_embedding+i];
}
// print __xlx_apatb_param_output_logits
for (size_t i = 0; i < __xlx_size_param_output_logits; ++i) {
((Byte<4>*)__xlx_apatb_param_output_logits)[i] = __xlx_output_logits__tmp_vec[__xlx_offset_param_output_logits+i];
}
// print __xlx_apatb_param_weight_token_embedding
for (size_t i = 0; i < __xlx_size_param_weight_token_embedding; ++i) {
((Byte<4>*)__xlx_apatb_param_weight_token_embedding)[i] = __xlx_weight_token_embedding__tmp_vec[__xlx_offset_param_weight_token_embedding+i];
}
// print __xlx_apatb_param_weight_attention_wq
for (size_t i = 0; i < __xlx_size_param_weight_attention_wq; ++i) {
((Byte<4>*)__xlx_apatb_param_weight_attention_wq)[i] = __xlx_weight_attention_wq__tmp_vec[__xlx_offset_param_weight_attention_wq+i];
}
// print __xlx_apatb_param_weight_attention_wk
for (size_t i = 0; i < __xlx_size_param_weight_attention_wk; ++i) {
((Byte<4>*)__xlx_apatb_param_weight_attention_wk)[i] = __xlx_weight_attention_wk__tmp_vec[__xlx_offset_param_weight_attention_wk+i];
}
// print __xlx_apatb_param_weight_attention_wv
for (size_t i = 0; i < __xlx_size_param_weight_attention_wv; ++i) {
((Byte<4>*)__xlx_apatb_param_weight_attention_wv)[i] = __xlx_weight_attention_wv__tmp_vec[__xlx_offset_param_weight_attention_wv+i];
}
// print __xlx_apatb_param_weight_attention_wo
for (size_t i = 0; i < __xlx_size_param_weight_attention_wo; ++i) {
((Byte<4>*)__xlx_apatb_param_weight_attention_wo)[i] = __xlx_weight_attention_wo__tmp_vec[__xlx_offset_param_weight_attention_wo+i];
}
// print __xlx_apatb_param_weight_ffn_w1
for (size_t i = 0; i < __xlx_size_param_weight_ffn_w1; ++i) {
((Byte<4>*)__xlx_apatb_param_weight_ffn_w1)[i] = __xlx_weight_ffn_w1__tmp_vec[__xlx_offset_param_weight_ffn_w1+i];
}
// print __xlx_apatb_param_weight_ffn_w2
for (size_t i = 0; i < __xlx_size_param_weight_ffn_w2; ++i) {
((Byte<4>*)__xlx_apatb_param_weight_ffn_w2)[i] = __xlx_weight_ffn_w2__tmp_vec[__xlx_offset_param_weight_ffn_w2+i];
}
// print __xlx_apatb_param_weight_ffn_w3
for (size_t i = 0; i < __xlx_size_param_weight_ffn_w3; ++i) {
((Byte<4>*)__xlx_apatb_param_weight_ffn_w3)[i] = __xlx_weight_ffn_w3__tmp_vec[__xlx_offset_param_weight_ffn_w3+i];
}
// print __xlx_apatb_param_weight_attention_norm
for (size_t i = 0; i < __xlx_size_param_weight_attention_norm; ++i) {
((Byte<4>*)__xlx_apatb_param_weight_attention_norm)[i] = __xlx_weight_attention_norm__tmp_vec[__xlx_offset_param_weight_attention_norm+i];
}
// print __xlx_apatb_param_weight_ffn_norm
for (size_t i = 0; i < __xlx_size_param_weight_ffn_norm; ++i) {
((Byte<4>*)__xlx_apatb_param_weight_ffn_norm)[i] = __xlx_weight_ffn_norm__tmp_vec[__xlx_offset_param_weight_ffn_norm+i];
}
// print __xlx_apatb_param_weight_final_norm
for (size_t i = 0; i < __xlx_size_param_weight_final_norm; ++i) {
((Byte<4>*)__xlx_apatb_param_weight_final_norm)[i] = __xlx_weight_final_norm__tmp_vec[__xlx_offset_param_weight_final_norm+i];
}
// print __xlx_apatb_param_k_cache
for (size_t i = 0; i < __xlx_size_param_k_cache; ++i) {
((Byte<4>*)__xlx_apatb_param_k_cache)[i] = __xlx_k_cache__tmp_vec[__xlx_offset_param_k_cache+i];
}
// print __xlx_apatb_param_v_cache
for (size_t i = 0; i < __xlx_size_param_v_cache; ++i) {
((Byte<4>*)__xlx_apatb_param_v_cache)[i] = __xlx_v_cache__tmp_vec[__xlx_offset_param_v_cache+i];
}
// print __xlx_apatb_param_cos_table
for (size_t i = 0; i < __xlx_size_param_cos_table; ++i) {
((Byte<4>*)__xlx_apatb_param_cos_table)[i] = __xlx_cos_table__tmp_vec[__xlx_offset_param_cos_table+i];
}
// print __xlx_apatb_param_sin_table
for (size_t i = 0; i < __xlx_size_param_sin_table; ++i) {
((Byte<4>*)__xlx_apatb_param_sin_table)[i] = __xlx_sin_table__tmp_vec[__xlx_offset_param_sin_table+i];
}
}
