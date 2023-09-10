#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cctype>
#include <pthread.h>

#include <tbb/concurrent_map.h>
#include <tbb/concurrent_queue.h>
#include <tbb/concurrent_set.h>
#include <tbb/concurrent_vector.h>

using namespace std;

uint32_t N, M=4;

//struct for X
tbb::concurrent_vector<string> X;
tbb::concurrent_set<uint32_t> X_done;
string get_file(){
    uint32_t f_id;
    do{
        f_id = rand() % N;
    }while(X_done.contains(f_id));
    X_done.insert(f_id);
    return X[f_id];
}

// struct for Y
template <typename T>
class FixedQueue : public tbb::concurrent_bounded_queue<T>{
private:
    T eof;

public:
    FixedQueue(uint32_t n){
        tbb::concurrent_bounded_queue<T>::set_capacity(n);
    }
    T pop(int *ret){
        T s;
        *ret = 0;
        tbb::concurrent_bounded_queue<T>::pop(s);

        if(s == eof){
            *ret = -1;
            tbb::concurrent_bounded_queue<T>::push(eof);
        }
        return s;
    }
    void end(){
        tbb::concurrent_bounded_queue<T>::push(eof);
    }
};
FixedQueue<string> Y(10);

tbb::concurrent_map<string,uint32_t> Z;

void* producer(void* arg) {
    // select a file randomly from X
    string file;
    file = get_file();

    // Reading selected file
    ifstream fin;
    fin.open(file);
    string line;
    while(getline(fin,line)){
        if(!line.empty())Y.push(line);
    }
    fin.close();

    pthread_exit(NULL);
}

// Function to remove punctuation from a word
string removePunctuation(const string& word) {
    string result = "";
    for (char c : word) {
        if (isalnum(c) || c == '\'') {
            result += c;
        }
    }
    return result;
}
void* consumer(void* arg) {

    while(true){
        // get a line from Y
        int i;
        string line = Y.pop(&i);
        if(i == -1)break;

        stringstream ss(line);
        string word;

        while(ss >> word){
            string cleanWord = removePunctuation(word);
            if (!cleanWord.empty()) {

                Z[cleanWord]++;

            }
        }
    }

    pthread_exit(NULL);
}

int main() {
    string infile;
    cout<<"Input file :";
    cin>>infile;

    ifstream fin;
    fin.open(infile);

    string line;
    while(getline(fin,line)){
        X.push_back(line);
        N++;
    }
    fin.close();

    if(M==0)M=N;

    pthread_attr_t attr;
    if (pthread_attr_init(&attr) == -1) {
        perror("error in pthread_attr_init");
        exit(1);
    }
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_t producers[N], consumers[M];
    int errcode;
    for (uint32_t id = 0; id < M; id++) {
        errcode = pthread_create(&consumers[id], &attr, consumer, NULL);
        if (errcode) {
            std::cout << "ERROR: return code from pthread_create() is " << errcode << "\n";
            exit(-1);
        }
    }
    for (uint32_t id = 0; id < N; id++) {
        errcode = pthread_create(&producers[id], &attr, producer, NULL);
        if (errcode) {
            std::cout << "ERROR: return code from pthread_create() is " << errcode << "\n";
            exit(-1);
        }
    }

    for (uint32_t id = 0; id < N; id++) {
        errcode = pthread_join(producers[id], NULL);
        if (errcode) {
            std::cout << "ERROR: return code from pthread_join() is " << errcode << "\n";
            exit(-1);
        }
    }
    Y.end();
    for (uint32_t id = 0; id < M; id++) {
        errcode = pthread_join(consumers[id], NULL);
        if (errcode) {
            std::cout << "ERROR: return code from pthread_join() is " << errcode << "\n";
            exit(-1);
        }
    }

    for(auto i : Z){
        cout<<i.first<<" "<<i.second<<endl;
    }

    pthread_exit(NULL);
}
