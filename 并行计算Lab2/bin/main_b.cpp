#pragma GCC optimize(3)

#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <queue>
#include <ctime>
#include <omp.h>

using namespace std;

#define INF 0x3f3f3f3f
struct edge {
    int to, cost;
};
typedef pair<int, int> P;
int n, m, count_, d_vec[14081999], source[999];
fstream Graph_file, Source_file;
vector<edge> G[14081999];
double runtime[999], totaltime, averagetime;

void dijkstra(int s) {
    priority_queue<P, vector<P>, greater<P> > que;
    fill(d_vec, d_vec + n, INF);
    d_vec[s] = 0;
    que.push(P(0, s));
    while (!que.empty()) {
        P p = que.top();
        que.pop();
        int v = p.second;
        if (d_vec[v] < p.first) continue;
        for (int i = 0; i < G[v].size(); i++) {
            edge e = G[v][i];
            if (d_vec[e.to] > d_vec[v] + e.cost) {
                d_vec[e.to] = d_vec[v] + e.cost;
                que.push(P(d_vec[e.to], e.to));
            }
        }
    }
}

void readfile() {
    char ch;
    for (int i = 0; i < 7; i++) {
        string sTmp;
        getline(Graph_file, sTmp);
    }
    int s, t, co;
    for (int i = 0; i < m; ++i) {
        Graph_file >> ch >> s >> t >> co;
        G[s - 1].push_back({t - 1, co});
    }
    string sTmp;
    while (Source_file >> ch && ch == 'c') { getline(Source_file, sTmp); }
    Source_file >> sTmp, Source_file >> sTmp, Source_file >> sTmp >> count_;
    //cout<<count_<<endl;
    while (Source_file >> ch && ch == 'c') { getline(Source_file, sTmp); }
    Source_file >> source[0];
    if (count_ > 1)
        for (int i = 1; i < count_; i++)
            Source_file >> ch >> source[i];
}

int main(int argc, char **argv) {
    clock_t startTime1, startTime2, endTime;
    startTime1 = clock();
    string Path(argv[2]);
    Path = "../data/USA-road-" + Path + ".";
    if (strcmp(argv[1], "CTR") == 0) {
        Path = Path + "CTR.";
        Graph_file.open(Path + "gr");
        Source_file.open(Path + "ss");
        n = 14081816, m = 34292496;
    } else if (strcmp(argv[1], "NE") == 0) {
        Path = Path + "NE.";
        Graph_file.open(Path + "gr");
        Source_file.open(Path + "ss");
        n = 1524453, m = 3897636;
    } else if (strcmp(argv[1], "NY") == 0) {
        Path = Path + "NY.";
        Graph_file.open(Path + "gr");
        Source_file.open(Path + "ss");
        n = 264346, m = 733846;
    }
    if (!Graph_file.is_open() || !Source_file.is_open()) {
        cout << "Error opening file" << endl;
        return -1;
    }
    cout << "Reading..." << endl;
    readfile();
    startTime2 = clock();
    cout << "The read time is: " << (double) (startTime2 - startTime1) / CLOCKS_PER_SEC << "s" << endl;
    //cin>>delta>>tnum;
    string kind(argv[1]);
    string fpath = "../bin/USA-road-" + (string) argv[2] + "." + kind + ".res";
    ofstream f1(fpath);
    f1 << "g " << n << " " << m << endl;
    count_ = 1;
    for (int i = 0; i < count_; i++) {
        startTime2 = clock();
        dijkstra(source[i]);
        endTime = clock();
        runtime[i] = (double) (endTime - startTime2) / CLOCKS_PER_SEC;
        f1 << "t " << runtime[i] << endl;
        totaltime += runtime[i];
    }
    averagetime = (double) totaltime / count_;
    cout << "The average run time is " << averagetime << "s." << endl;
    cout << "The result is successfully saved in '../bin/." << endl;
    return 0;
}