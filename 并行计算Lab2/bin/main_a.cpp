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
#define N 14081999
int delta, tnum, n, m, maxbucket, cnt, tent[N], source[999];
struct edge {
    int to, cost;
};
struct re_queue {
    int v, w;
};
fstream Graph_file, Source_file;
vector<edge> Graph[N];
vector<int> bucket[N], stack;
vector<re_queue> release_queue;
double totaltime, averagetime, runtime[999];

bool bempty() {
    for (int i = 0; i <= maxbucket; i++)
        if (!bucket[i].empty())
            return false;
    return true;
}

void relax(int w, int d) {
    if (d < tent[w]) {
        if (tent[w] != INF) {
            vector<int>::iterator res = find(bucket[tent[w] / delta].begin(), bucket[tent[w] / delta].end(), w);
            if (res != bucket[tent[w] / delta].end())
                bucket[tent[w] / delta].erase(res);
        }
        bucket[d / delta].push_back(w);
        if (d / delta > maxbucket) maxbucket = d / delta;
        tent[w] = d;
    }
}

void deltastepping(int s) {
    maxbucket = 0;
    for (int i = 0; i < n; i++)
        tent[i] = INF;
    relax(s, 0);
    int j = 0;
    omp_set_num_threads(4);
    while (!bempty()) {
        stack.clear();
        while (!bucket[j].empty()) {
            release_queue.clear();
#pragma omp for
            for (int i = 0; i < bucket[j].size(); i++) {
                int vv = bucket[j][i];
                for (int k = 0; k < Graph[vv].size(); k++)
                    if (Graph[vv][k].cost <= delta) {
                        re_queue r;
                        r.v = Graph[vv][k].to;
                        r.w = tent[vv] + Graph[vv][k].cost;
                        release_queue.push_back(r);
                    }
                stack.push_back(vv);
            }
            bucket[j].clear();
#pragma omp for
            for (int i = 0; i < release_queue.size(); i++)
                relax(release_queue[i].v, release_queue[i].w);
        }
        release_queue.clear();
#pragma omp for
        for (int i = 0; i < stack.size(); i++) {
            int vv = stack[i];
            for (int k = 0; k < Graph[vv].size(); k++)
                if (Graph[vv][k].cost > delta) {
                    re_queue r;
                    r.v = Graph[vv][k].to;
                    r.w = tent[vv] + Graph[vv][k].cost;
                    release_queue.push_back(r);
                }
        }
#pragma omp for
        for (int i = 0; i < release_queue.size(); i++)
            relax(release_queue[i].v, release_queue[i].w);
        j++;
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
        Graph[s - 1].push_back({t - 1, co});
    }
    string sTmp;
    while (Source_file >> ch && ch == 'c') { getline(Source_file, sTmp); }
    Source_file >> sTmp, Source_file >> sTmp, Source_file >> sTmp >> cnt;
    //cout<<cnt<<endl;
    while (Source_file >> ch && ch == 'c') { getline(Source_file, sTmp); }
    Source_file >> source[0];
    if (cnt > 1)
        for (int i = 1; i < cnt; i++)
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
        delta = 4000;
    } else if (strcmp(argv[1], "NE") == 0) {
        Path = Path + "NE.";
        Graph_file.open(Path + "gr");
        Source_file.open(Path + "ss");
        n = 1524453, m = 3897636;
        delta = 3000;
    } else if (strcmp(argv[1], "NY") == 0) {
        Path = Path + "NY.";
        Graph_file.open(Path + "gr");
        Source_file.open(Path + "ss");
        n = 264346, m = 733846;
        delta = 10;
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
    cnt = 1;
    for (int i = 0; i < cnt; i++) {
        startTime2 = clock();
        deltastepping(source[i]);
        endTime = clock();
        runtime[i] = (double) (endTime - startTime2) / CLOCKS_PER_SEC;
        f1 << "t " << runtime[i] << endl;
        totaltime += runtime[i];
    }
    averagetime = (double) totaltime / cnt;
    cout << "The average run time is " << averagetime << "s." << endl;
    cout << "The result is successfully saved in '../bin/." << endl;
    return 0;
}
