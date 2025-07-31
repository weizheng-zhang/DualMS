//MIT License
//
//Copyright(c) 2020 Zheng Jiaqi @NUSComputing
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files(the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions :
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <string>
#include <queue>

#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp> 

#include "gCVT\gCVT.h"

#include <chrono> // calculate time

// Input parameters
int fboSize = 256;
int nVertices = 100;

int depth = 1;
int maxIter = 10000;

int laplace_iter = 0;
float lambda_factor = 0.1;

#define TOID(x, y, z, w)    ((z) * (w) * (w) + (y) * (w) + (x))

// Global Vars
int* inputPoints, * inputVoronoi, * outputVoronoi;
float* inputDensity;

// Initialization
void initialization()
{
    gCVTInitialization(fboSize);

    outputVoronoi = (int*)malloc(fboSize * fboSize * fboSize * sizeof(int));
    inputDensity = (float*)malloc(fboSize * fboSize * fboSize * sizeof(float));
}

// Deinitialization
void deinitialization()
{
    gCVTDeinitialization();

    free(inputDensity);
    free(outputVoronoi);
}

void initializeDensity()
{
	// open input file
    freopen("./U_tube.txt", "r", stdin);
    for (int i = 0; i < fboSize; ++i) {
        for (int j = 0; j < fboSize; ++j) {
            for (int k = 0; k < fboSize; ++k) {
                scanf("%f", &inputDensity[TOID(i, j, k, fboSize)]);
            }
        }
    }
    fclose(stdin);
    freopen("CONIN$", "r", stdin);
}

// Run the tests
void runTests()
{
    initializeDensity();

    srand(time(NULL));
    gCVT(inputDensity, nVertices, outputVoronoi, depth, maxIter);

}

// Divide2Set
void update_lable(std::vector<int>& lable, std::vector<int>& set1, std::vector<int>& set2) {
    for (auto id : set1) {
        lable[id] = 1;
    }
    for (auto id : set2) {
        lable[id] = 2;
    }
}

int get_score(std::vector<std::vector<std::pair<int, int>>>& g, std::vector<int>& set1, std::vector<int>& set2, std::vector<int>& lable) {
    int score = 0;
    for (int i = 0; i < g.size(); i++) {
        for (auto p : g[i]) {
            int j = p.first, w = p.second;
            if (i > j) continue;
            if (lable[i] != lable[j]) score += w;
        }
    }
    return score;
}

void copy_vector(std::vector<int>& v, const std::vector<int>& copy) {
    v.resize(copy.size());
    for (int i = 0; i < copy.size(); i++) {
        v[i] = copy[i];
    }
}

void dfs_check_union(int u, std::vector<std::vector<std::pair<int, int>>>& g, std::vector<int>& lable, int id, std::vector<int>& vis) {
    vis[u] = id;
    for (auto p : g[u]) {
        int v = p.first;
        if (lable[v] != id || vis[v]) continue;
        dfs_check_union(v, g, lable, id, vis);
    }
}

bool check_union(std::vector<std::vector<std::pair<int, int>>>& g, int set1, int set2, std::vector<int>& lable) {
    std::vector<int>vis;
    vis.resize(g.size(), 0); // 0:未访问 1:在set1中 2:在set2中
    dfs_check_union(set1, g, lable, 1, vis);
    dfs_check_union(set2, g, lable, 2, vis);
    int num = 0;
    for (auto v : vis) {
        if (v != 0) num++;
    }
    return num == vis.size() - 1;
}

void divide2Set(std::vector<std::tuple<float, float, float>>points, std::vector<std::vector<std::pair<int, int>>>& graph, std::vector<int>& set1, std::vector<int>& set2, std::vector<int>& isBoundry) {
    std::vector<int>res1, res2;
    int max_score = -0x3f3f3f3f;

    int g_size = graph.size();

    for (int t = g_size / 2; t <= g_size / 2; t++) {
        std::vector<int>cur1, cur2;
        std::vector<int> vis;
        vis.resize(graph.size());
        for (int i = 1; i < graph.size(); i++) {
            if (get<1>(points[i - 1]) < 0.5)cur1.push_back(i);
            else cur2.push_back(i);
        }

        for (auto it : cur1) {
            vis[it] = 1;
        }
        for (auto it : cur2) {
            vis[it] = 2;
        }
        int cur_score = get_score(graph, cur1, cur2, vis);
        if (max_score < cur_score) {
            max_score = cur_score;
            copy_vector(res1, cur1);
            copy_vector(res2, cur2);
        }
        std::cout << "set1's size: " + std::to_string(t) + ", max_score: " + std::to_string(max_score) + ", cur_score: " + std::to_string(cur_score) << std::endl;

    }
    std::cout << "set1'size: " + std::to_string(res1.size()) + ", set2's size: " + std::to_string(res2.size()) << std::endl;

    std::cout << "start K_L_operation" << std::endl;
    std::cout << "graph.size()" << graph.size() << std::endl;
    std::vector<int> vis;
    vis.resize(graph.size());
    for (auto it : res1) {
        vis[it] = 1;
    }
    for (auto it : res2) {
        vis[it] = 2;
    }
    int cur_score = get_score(graph, res1, res2, vis);

    for (int epoch = 1; epoch <= graph.size(); epoch++) {
        int max_id = -1, score = 0;

        // Preprocessing determines whether the number of points of this point and its neighbors is greater than 2
        std::vector<int>degree(graph.size()); // How many points have the same label as each point
        for (int i = 1; i < graph.size(); i++) {
            degree[i] = 0;
            for (auto p : graph[i]) {
                int to = p.first;
                if (vis[i] == vis[to]) degree[i]++;
            }
        }
        std::vector<int>can_exchange(graph.size()); // whether the point interchangeable（whether the degrees of this point and its neighbors greater than 2）
        for (int i = 1; i < graph.size(); i++) {
            can_exchange[i] = ((graph[i].size() - degree[i]) > 1); // Current point's degree must be greater than 2, and the degree transferred to the other party's set must also be greater than or equal to 2
            for (auto p : graph[i]) {
                int to = p.first;
                if (vis[i] == vis[to]) {
                    if (degree[to] <= 2) can_exchange[i] = 0;
                }
            }
        }

        for (int from = 1; from < vis.size(); from++) {

            if (!can_exchange[from]) continue;// The degree of the current point or its neighbor is less than 2 and cannot be exchanged.
            int label = vis[from];
            for (auto p : graph[from]) {
                int to = p.first, w = p.second;
                if (vis[to] == label) cur_score += w;
                else cur_score -= w;
            }
            vis[from] = 3 - vis[from]; //update from label
            if (cur_score > score) {
                int u = -1, v = -1;
                for (int i = 1; i < vis.size(); i++) { 
                    if (u == -1 && vis[i] == 1) {
                        u = i;
                    }
                    else if (v == -1 && vis[i] == 2) {
                        v = i;
                    }
                    if (u != -1 && v != -1) break;
                }
                if (check_union(graph, u, v, vis)) {
                    score = cur_score;
                    max_id = from;
                }
            }
            // recover cur_score
            for (auto p : graph[from]) {
                int to = p.first, w = p.second;
                if (vis[to] == label) cur_score -= w;
                else cur_score += w;
            }
            vis[from] = 3 - vis[from]; //recover from label
        }
        if (max_id == -1) {
            std::cout << "max_id == -1！！ epoch: " << epoch << " max_id: " << max_id << " score: " << score << std::endl;
            break;
        }

		// transfer max_id to the other set
        for (auto p : graph[max_id]) {
            int to = p.first, w = p.second;
            if (vis[to] == vis[max_id]) cur_score += w;
            else cur_score -= w;
        }
        vis[max_id] = 3 - vis[max_id]; // update vis
        max_score = std::max(max_score, cur_score);
        if (epoch % 10 == 0) {
            std::cout << "epoch:" << " " << epoch << " max_score:" << max_score << std::endl;
        }
    }

    // update res1,res2
    res1.clear(); res2.clear();
    for (int id = 0; id < vis.size(); id++) {
        if (vis[id] == 1) res1.push_back(id);
        else if (vis[id] == 2)res2.push_back(id);
    }
    int edge_num = 0;
    for (int i = 0; i < graph.size(); i++) {
        edge_num += graph[i].size();
    }
    edge_num /= 2;
    int num_divide = 0, num_divide2 = 0;
    for (int i = 0; i < graph.size(); i++) {
        for (auto p : graph[i]) {
            int j = p.first;
            if (vis[i] == vis[j]) num_divide++;
        }
    }

    std::cout << "set1's size: " + std::to_string(res1.size()) + ", set2's size: " + std::to_string(res2.size()) + ", max_score: " + std::to_string(max_score) + ", edge num: " + std::to_string(edge_num)
        + ", E1+E2: " + std::to_string(num_divide / 2) + ", post solve score:" + std::to_string(edge_num - num_divide / 2) << std::endl;

    copy_vector(set1, res1);
    copy_vector(set2, res2);
    return;
}


float dot_product(const std::tuple<float, float, float>& v1, const std::tuple<float, float, float>& v2) {
    return std::get<0>(v1) * std::get<0>(v2) + std::get<1>(v1) * std::get<1>(v2) + std::get<2>(v1) * std::get<2>(v2);
}

float magnitude(const std::tuple<float, float, float>& v) {
    return std::sqrt(std::get<0>(v) * std::get<0>(v) + std::get<1>(v) * std::get<1>(v) + std::get<2>(v) * std::get<2>(v));
}

float angle_between_vectors(const std::tuple<float, float, float>& v1, const std::tuple<float, float, float>& v2) {
    float dot = dot_product(v1, v2);
    float mag1 = magnitude(v1);
    float mag2 = magnitude(v2);
    float cos_theta = dot / (mag1 * mag2);
	return std::acos(cos_theta); // returns the angle in radians
}

std::tuple<float, float, float> subtract_tuples(const std::tuple<float, float, float>& t1, const std::tuple<float, float, float>& t2) {
    return std::make_tuple(
        std::get<0>(t1) - std::get<0>(t2),
        std::get<1>(t1) - std::get<1>(t2),
        std::get<2>(t1) - std::get<2>(t2)
    );
}

void flow_field(std::vector<std::tuple<float, float, float>>& points, std::vector<std::vector<std::pair<int, int>>>& graph, std::vector<int>& start) {
    std::tuple<float, float, float> circle_center = std::make_tuple(0.35, 0.5, 0.15);
    for (int i = 1; i < graph.size(); i++) {
        for (auto& p : graph[i]) {
            int to = p.first;
            std::tuple<float, float, float> circle_dir = subtract_tuples(circle_center, points[to - 1]);
            auto [x, y, z] = circle_dir;
            circle_dir = std::make_tuple(y, -x, 0);
            if (std::get<0>(points[i - 1]) > 0.3) {
                circle_dir = std::make_tuple(1, 0, 0);
            }
            std::tuple<float, float, float> edge_dir = subtract_tuples(points[i - 1], points[to - 1]);
            float angle = angle_between_vectors(circle_dir, edge_dir);
            float pi = 3.1415926;
            int val = 1;
            if (angle <= pi / 4 || angle >= pi / 4 * 3) {
                p.second = val;
            }
            else {
                p.second = val;
            }
        }
    }
}

// Visualization
void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}
void drawPoints(std::vector<std::tuple<float, float, float>>& points) {
    glBegin(GL_POINTS);
    for (const auto& point : points) {
        glVertex3f(get<0>(point), get<1>(point), get<2>(point));
    }
    glEnd();
}

bool isDragging = false;
bool isRightDragging = false;
double lastMouseX, lastMouseY;
float angleX = 0.0f, angleY = 0.0f;
float zoom = -5.0f;
float translateX = 0.0f, translateY = 0.0f;

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            isDragging = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        }
        else if (action == GLFW_RELEASE) {
            isDragging = false;
        }
    }
    else if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            isRightDragging = true;
            glfwGetCursorPos(window, &lastMouseX, &lastMouseY);
        }
        else if (action == GLFW_RELEASE) {
            isRightDragging = false;
        }
    }
}

void cursorPositionCallback(GLFWwindow* window, double xpos, double ypos) {
    if (isDragging) {
        double deltaX = xpos - lastMouseX;
        double deltaY = ypos - lastMouseY;
        angleX += deltaY * 0.1f;
        angleY += deltaX * 0.1f;
        lastMouseX = xpos;
        lastMouseY = ypos;
    }
    else if (isRightDragging) {
        double deltaX = xpos - lastMouseX;
        double deltaY = ypos - lastMouseY;
        translateX += deltaX * 0.01f;
        translateY -= deltaY * 0.01f;
        lastMouseX = xpos;
        lastMouseY = ypos;
    }
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    zoom += yoffset * 0.1f;
}

// define a 3D point structure
struct Point3D {
    float x, y, z;
};

// create a vector of edges for the cube
std::vector<Point3D> cubeEdges = {
    {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 1.0f},
    {0.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 1.0f},
    {1.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 1.0f},

    {1.0f, 1.0f, 0.0f}, {0.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
    {0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 0.0f},

    {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f, 0.0f},
    {0.0f, 1.0f, 1.0f}, {0.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f},
    {1.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}
};
void drawCubeEdges() {
    glBegin(GL_LINES);
    for (const auto& edge : cubeEdges) {
        glVertex3f(edge.x, edge.y, edge.z);
    }
    glEnd();
}

int main(int argc, char** argv)
{
	auto start1 = std::chrono::high_resolution_clock::now(); // calcate start time
    initialization();

    runTests();

    std::cout << "runTests Done!" << std::endl;
    // get points
    int point_id = 0;
    std::map<std::tuple<int, int, int>, int> point_map;
    std::map<int, std::tuple<float, float, float>> id_map;
    std::vector<std::tuple<float, float, float>>points;
    std::vector<std::vector<std::pair<int, int>>> graph(nVertices + 1);
    std::map<std::pair<int, int>, int>weighted_graph;
    std::vector<int>isBoundry(nVertices + 1, 0);
    for (int i = 0; i < fboSize; ++i) {
        for (int j = 0; j < fboSize; ++j) {
            for (int k = 0; k < fboSize; ++k) {
                if (outputVoronoi[TOID(i, j, k, fboSize)] != MARKER) {
                    int x, y, z;
                    DECODE(outputVoronoi[TOID(i, j, k, fboSize)], x, y, z);
                    int val = point_map[{x, y, z}];
                    if (!val) {
                        points.push_back({ x * 1.0 / fboSize, y * 1.0 / fboSize, z * 1.0 / fboSize });
                        point_map[{ x, y, z }] = ++point_id;
                        id_map[point_id] = { x * 1.0 / fboSize, y * 1.0 / fboSize, z * 1.0 / fboSize };
                    }
                }
            }
        }
    }
    std::cout << "get points Done!" << std::endl;

    // get Delaunay Triangulation
    const int dx[] = { 0,0,1,-1,0,0 };
    const int dy[] = { 1,-1,0,0,0,0 };
    const int dz[] = { 0,0,0,0,1,-1 };
    for (int i = 0; i < fboSize; ++i) {
        for (int j = 0; j < fboSize; ++j) {
            for (int k = 0; k < fboSize; ++k) {
                if (inputDensity[TOID(i, j, k, fboSize)] < 1) continue;
                if (outputVoronoi[TOID(i, j, k, fboSize)] == MARKER) continue;
                for (int step = 0; step < 6; step++) {
                    int x = i + dx[step], y = j + dy[step], z = k + dz[step];
                    if (x < 0 || x >= fboSize) {
                        if (x < fboSize / 2) continue;
                        int x1, y1, z1;
                        DECODE(outputVoronoi[TOID(i, j, k, fboSize)], x1, y1, z1);
                        int id1 = point_map[{x1, y1, z1}];
                        isBoundry[id1] = 1; // boundary
                    }
                    if (x < 0 || x >= fboSize || y < 0 || y >= fboSize || z < 0 || z >= fboSize) {
                        continue;
                    }
                    if (outputVoronoi[TOID(i, j, k, fboSize)] != MARKER && outputVoronoi[TOID(x, y, z, fboSize)] != MARKER) {
                        if (inputDensity[TOID(x, y, z, fboSize)] < 1) continue;
                        if (outputVoronoi[TOID(i, j, k, fboSize)] == outputVoronoi[TOID(x, y, z, fboSize)]) continue;
                        int x1, y1, z1, x2, y2, z2;
                        DECODE(outputVoronoi[TOID(i, j, k, fboSize)], x1, y1, z1);
                        DECODE(outputVoronoi[TOID(x, y, z, fboSize)], x2, y2, z2);
                        int id1 = point_map[{x1, y1, z1}];
                        int id2 = point_map[{x2, y2, z2}];
                        weighted_graph[{id1, id2}]++;
                        weighted_graph[{id2, id1}]++;
                    }
                }
            }
        }
    }
    std::cout << "get graph Done!" << std::endl;


    for (auto it : weighted_graph) {
        auto u = it.first.first, v = it.first.second;
        auto w = it.second;
        if (w >= 1) graph[u].push_back({ v,w });
    }

    // 生成流场图
    std::vector<int>start;
    for (int i = 1; i < isBoundry.size(); i++) {
        if (isBoundry[i] == 0) continue;
        auto [x, y, z] = points[i - 1];
        if (y < 0.5) start.push_back(i);
    }

    flow_field(points, graph, start);
	auto end1 = std::chrono::high_resolution_clock::now(); // calcate end time
	std::chrono::duration<double> elapsed1 = end1 - start1; // calcate time difference (seconds)
    std::cout << "The time of stage 1: " << elapsed1.count() << " 秒" << std::endl;

	// dual skeleton optimization
    std::vector<int> set1, set2;
    divide2Set(points, graph, set1, set2, isBoundry);
    std::vector<int>lable(points.size() + 1);

    update_lable(lable, set1, set2);

    set1.clear(); set2.clear(); // update set1, set2
    for (int i = 1; i < lable.size(); i++) {
        if (lable[i] == 1) set1.push_back(i);
        else set2.push_back(i);
    }

    puts("generate done!");
    std::cout << set1.size() << " " << set2.size() << " " << lable.size() << std::endl;
	auto end2 = std::chrono::high_resolution_clock::now(); // calcate end time
	std::chrono::duration<double> elapsed2 = end2 - end1; // calculate time difference (seconds)
    std::cout << "The time of stage 2: " << elapsed2.count() << " 秒" << std::endl;

	// save points and graph
    std::cout << "save file" << std::endl;
    freopen("density.txt", "w", stdout);
    std::cout << points.size() << std::endl;
    for (auto p : points) {
        std::cout << get<0>(p) << " " << get<1>(p) << " " << get<2>(p) << std::endl;
    }
    //
    std::cout << set1.size() << std::endl;
    for (auto id : set1) {
        std::cout << id - 1 << " ";
    }
    std::cout << std::endl;
    int num = 0;
    for (int i = 1; i < graph.size(); i++) {
        if (std::find(set1.begin(), set1.end(), i) == set1.end()) continue;
        for (auto p : graph[i]) {
            int to = p.first;
            if (std::find(set1.begin(), set1.end(), to) == set1.end()) continue;
            num++;
        }
    }
    std::cout << num << std::endl;
    for (int i = 1; i < graph.size(); i++) {
        if (std::find(set1.begin(), set1.end(), i) == set1.end()) continue;
        for (auto p : graph[i]) {
            int to = p.first;
            if (std::find(set1.begin(), set1.end(), to) == set1.end()) continue;
            std::cout << i - 1 << " " << to - 1 << std::endl;
        }
    }
    std::cout << set2.size() << std::endl;
    for (auto id : set2) {
        std::cout << id - 1 << " ";
    }
    std::cout << std::endl;
    num = 0;
    for (int i = 1; i < graph.size(); i++) {
        if (std::find(set2.begin(), set2.end(), i) == set2.end()) continue;
        for (auto p : graph[i]) {
            int to = p.first;
            if (std::find(set2.begin(), set2.end(), to) == set2.end()) continue;
            num++;
        }
    }
    std::cout << num << std::endl;
    for (int i = 1; i < graph.size(); i++) {
        if (std::find(set2.begin(), set2.end(), i) == set2.end()) continue;
        for (auto p : graph[i]) {
            int to = p.first;
            if (std::find(set2.begin(), set2.end(), to) == set2.end()) continue;
            std::cout << i - 1 << " " << to - 1 << std::endl;
        }
    }
	// save cut edges
    num = 0;
    for (int i = 1; i < graph.size(); i++) {
        if (std::find(set1.begin(), set1.end(), i) == set1.end()) continue;
        for (auto p : graph[i]) {
            int to = p.first;
            if (std::find(set2.begin(), set2.end(), to) == set2.end()) continue;
            num++;
        }
    }
    std::cout << num << std::endl;
    for (int i = 1; i < graph.size(); i++) {
        if (std::find(set1.begin(), set1.end(), i) == set1.end()) continue;
        for (auto p : graph[i]) {
            int to = p.first;
            if (std::find(set2.begin(), set2.end(), to) == set2.end()) continue;
            std::cout << i - 1 << " " << to - 1 << std::endl;
        }
    }
    fclose(stdout);

    freopen("CON", "a", stdout); 

    std::cout << "Done!" << std::endl;

	// visualization
	// initialize GLFW
    if (!glfwInit()) {
        return -1;
    }

	// create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(1920, 1080, "3D Points Visualization", NULL, NULL);
    if (!window) {
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);
    glEnable(GL_DEPTH_TEST);
    glPointSize(5.0);

    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

	// set up the viewport
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorPositionCallback);
    glfwSetScrollCallback(window, scrollCallback);

	// main loop
    while (!glfwWindowShouldClose(window)) {
        processInput(window);

		// clear the screen
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// set up the camera
        glm::mat4 view = glm::translate(glm::mat4(1.0f), glm::vec3(translateX, translateY, zoom));
        glm::mat4 projection = glm::perspective(glm::radians(45.0f), 1920.0f / 1080.0f, 0.1f, 100.0f);
        glm::mat4 model = glm::rotate(glm::mat4(1.0f), glm::radians(angleX), glm::vec3(1.0f, 0.0f, 0.0f));
        model = glm::rotate(model, glm::radians(angleY), glm::vec3(0.0f, 1.0f, 0.0f));

        glm::mat4 mvp = projection * view * model;
        glLoadMatrixf(glm::value_ptr(mvp));

		// draw points
        glColor3f(1.0, 1.0, 1.0);
        glEnable(GL_POINT_SMOOTH);
		glPointSize(5.0f); // set point size
        glBegin(GL_POINTS);
        int idx = 0;
        for (const auto& point : points) {
            idx++;
            if (lable[idx] == 1) glColor3f(1.0, 0.0, 0.0);
            else glColor3f(0.0, 0.0, 1.0);

            glVertex3f(get<0>(point), get<1>(point), get<2>(point));
        }
        glEnd();

		// draw edges
        glColor3f(1.0, 1.0, 0.0);
        for (int i = 1; i < graph.size(); i++) {
            for (auto p : graph[i]) {
                int j = p.first;

                if (lable[i] == lable[j]) {
                    if (lable[i] == 1) glColor3f(1.0, 0.0, 0.0);
                    else glColor3f(0.0, 0.0, 1.0);
                }
                else continue;
                glBegin(GL_LINES);
                auto p1 = points[i - 1], p2 = points[j - 1];
                glVertex3f(get<0>(p1), get<1>(p1), get<2>(p1));
                glVertex3f(get<0>(p2), get<1>(p2), get<2>(p2));
                glEnd();
            }
        }

        glColor3f(0.0, 0.0, 0.0);

		// draw cube edges
        glLineWidth(2.5f);
        drawCubeEdges();
        glLineWidth(1.0f);

		// swapping buffers and polling events
        glfwSwapBuffers(window);
        glfwPollEvents();

    }

    glfwTerminate();

    deinitialization();

    return 0;
}