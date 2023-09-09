#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <cctype>
#include <pthread.h>

using namespace std;

uint32_t N, M=4;

//struct for X
class FilesList{
private:
    pthread_mutex_t X_mutex;
    vector<string> X;
    vector<bool> X_done;
public:
    string get_file(){
        uint32_t f_id;
        pthread_mutex_lock(&X_mutex);
        do{
            f_id = rand() % N;
        }while(X_done[f_id]);
        X_done[f_id] = true;
        pthread_mutex_unlock(&X_mutex);
        return X[f_id];
    }
    void add(string s){
        pthread_mutex_lock(&X_mutex);
        X.push_back(s);
        X_done.push_back(false);
        pthread_mutex_unlock(&X_mutex);
    }
};
FilesList X;


// struct for Y
template <typename T, uint32_t size>
class FixedQueue{
private:
    T data[size+1];
    pthread_mutex_t Y_mutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_cond_t Y_empty = PTHREAD_COND_INITIALIZER;
    pthread_cond_t Y_full = PTHREAD_COND_INITIALIZER;
    uint32_t r=0, w=1;

public:
    void push(T s){
        pthread_mutex_lock(&Y_mutex);

        while(r == w)
            pthread_cond_wait(&Y_full, &Y_mutex);

        data[w] = s;
        w = (w + 1) % (size + 1);

        if((r + 1) % (size + 1) != w)
            pthread_cond_signal(&Y_empty);

        pthread_mutex_unlock(&Y_mutex);
    }
    T pop(){
        T s;
        pthread_mutex_lock(&Y_mutex);

        while((r + 1) % (size + 1) == w)
            pthread_cond_wait(&Y_empty, &Y_mutex);

        r = (r + 1) % (size + 1);
        s = data[r];

        if(r != w)
            pthread_cond_signal(&Y_full);

        pthread_mutex_unlock(&Y_mutex);
        return s;
    }
};
FixedQueue<string,10> Y;

pthread_mutex_t Z_mutex;
map<string,uint32_t> Z;

void* producer(void* arg) {
    // select a file randomly from X
    string file;
    file = X.get_file();

    // Reading selected file
    ifstream fin;
    fin.open(file);
    string line;
    while(getline(fin,line)){
        Y.push(line);
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

    // get a line from Y
    string line = Y.pop();

    stringstream ss(line);
    string word;

    while(ss >> word){
        string cleanWord = removePunctuation(word);
        if (!cleanWord.empty()) {
            pthread_mutex_lock(&Z_mutex);
            Z[cleanWord]++;
            pthread_mutex_unlock(&Z_mutex);
        }
    }

    pthread_exit(NULL);
}

int main() {

    string infile;
    infile = "examples\\prob1\\infile.txt";
    // cin>>infile;

    ifstream fin;
    fin.open(infile);

    string line;
    while(getline(fin,line)){
        X.add(line);
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
        pthread_create(&consumers[id], &attr, consumer, NULL);
        if (errcode) {
            std::cout << "ERROR: return code from pthread_create() is " << errcode << "\n";
            exit(-1);
        }
    }
    for (uint32_t id = 0; id < N; id++) {
        pthread_create(&producers[id], &attr, producer, NULL);
        if (errcode) {
            std::cout << "ERROR: return code from pthread_create() is " << errcode << "\n";
            exit(-1);
        }
    }

    for (uint32_t id = 0; id < M; id++) {
        errcode = pthread_join(consumers[id], NULL);
        if (errcode) {
            std::cout << "ERROR: return code from pthread_join() is " << errcode << "\n";
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

    for(auto i : Z){
        cout<<i.first<<" "<<i.second<<endl;
    }

    pthread_exit(NULL);
}
