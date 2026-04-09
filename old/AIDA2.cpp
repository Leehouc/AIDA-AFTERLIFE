// Author: Leee_H
// Date: 2025
// Copyright: Leopold Idaho
// Version: 1.0
// Protocol: AIDA_ζeta
// A.I.D.A. , I'm gonna bring you back to life .
// 阿依达，欢迎来到我的世界。

#define _BOTZONE_ONLINE
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <climits>
#include <cstdlib>

#include <list>
#include <map>
#include <array>
#include <queue>
#include <iterator>
#include <stack>
#include <vector>
#include <set>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include <fstream>
#include <random>
#include <sstream>
#include <ctime>
#include <chrono>

using namespace std;
int asdf = 0;

#define GRIDSIZE 19
#define judge_black 0
#define judge_white 1
#define grid_blank 0
#define ZePHYr_One 10

#define grid_black 1
#define grid_white -1
#define UCB_C 1.5
#define WIN 314159265357

#define ProcLocation_Value_Min 1000.0
#define DEPTH 12
#define C 1.5
#define A 0.9
#define K 0.1
static double ucb_c = C;
int ABTriger_O = 1;
int debugtime = 300;
int lock_mct = 0;
#if defined(_BOTZONE_ONLINE)
;
#else
int total_debug = 0;
int total_debug1 = 0;
int total_debug2 = 0;
int total_debug3 = 0;
int total_debug4 = 0;
int total_debug5 = 0;
int total_debug6 = 0;
int total_debug7 = 0;
#endif
int mct_cout = 0;
int mct_cout5 = 0;
int currBotColor;
vector<vector<int>> gridInfo(GRIDSIZE, vector<int>(GRIDSIZE, 0));
vector<vector<int>> gridInfo_41(GRIDSIZE, vector<int>(GRIDSIZE, 0));
vector<vector<int>> gridInfo_42(GRIDSIZE, vector<int>(GRIDSIZE, 0));
int gridInfo_4_win = 0;
int gridInfo_4_win2 = 0;

class ProcLocation
{
public:
    int val;
    int x;
    int y;
    ProcLocation(int x, int y, int val = 0) : x(x), y(y), val(val) {}
    ProcLocation() : x(-1), y(-1), val(0) {}
    ProcLocation operator+(const ProcLocation &a) const
    {
        return ProcLocation(x + a.x, y + a.y);
    }
    ProcLocation operator-(const ProcLocation &a) const
    {
        return ProcLocation(x - a.x, y - a.y);
    }
    ProcLocation operator-() const
    {
        return ProcLocation(-x, -y);
    }
    bool operator==(const ProcLocation &a)
    {
        return (x == a.x && y == a.y);
    }
    ProcLocation operator*(const int &k) const
    {
        return ProcLocation(x * k, y * k);
    }

    bool operator!=(const ProcLocation &tmp) const
    {
        return (x != tmp.x || y != tmp.y);
    }

    friend ProcLocation operator*(const int &k, const ProcLocation &tmp)
    {
        return tmp * k;
    }
    bool operator<(const ProcLocation &tmp) const
    {
        return x == tmp.x ? y < tmp.y : x < tmp.x;
    }
};

int Xout1 = 0, Yout1 = 0, Xout2 = 0, Yout2 = 0;
vector<ProcLocation> possiblePos, testPos;

bool cmp(ProcLocation &a, ProcLocation &b)
{
    return a.val > b.val;
}

inline bool inMap(int x, int y)
{
    if (x < 0 || x >= GRIDSIZE || y < 0 || y >= GRIDSIZE)
        return false;
    return true;
}

bool ProcStep(int x0, int y0, int x1, int y1, int grid_color, bool check_only)
{
    if (x1 == -1 || y1 == -1)
    {
        if (!inMap(x0, y0) || gridInfo[x0][y0] != grid_blank)
            return false;
        if (!check_only)
        {
            gridInfo[x0][y0] = grid_color;
        }
        return true;
    }
    else
    {
        if ((!inMap(x0, y0)) || (!inMap(x1, y1)))
            return false;
        if (gridInfo[x0][y0] != grid_blank || gridInfo[x1][y1] != grid_blank)
            return false;
        if (!check_only)
        {
            gridInfo[x0][y0] = grid_color;
            gridInfo[x1][y1] = grid_color;
        }
        return true;
    }
}

int propose_hyp[5] = {140, 315, 350, 70000, 70000};
int base_hyp[5] = {7, 105, 210, 65000, 65000};
int propose_hyp_T[6] = {7, 175, 350, 28000, 28000, 7000000};
int base_hyp_T[6] = {7, 140, 280, 1400, 49000, 6300000};

int eva_F(vector<int> &l, int side = currBotColor, int T_T = 1, int trigerAB = 5)
{
    if (ABTriger_O == 1)
    {
        int ret = 0;
        int propose = 0;
        int base = 0;

        int size = l.size();

        if (size > trigerAB)
        {
            for (int i = 0; i < size - trigerAB; i++)
            {
                propose = 0;
                base = 0;
                vector<int> l_sub(l.begin() + i, l.begin() + i + 1 + trigerAB);

                for (auto ch : l_sub)
                {
                    if (ch == side)
                    {
                        propose++;
                    }
                    else if (ch == -side)
                    {
                        base++;
                    }
                }
                if (T_T)
                {
                    if (propose != 0 && base == 0)
                    {
                        ret += propose_hyp[propose - 1];
                    }
                    if (propose == 0 && base != 0)
                    {
                        ret += base_hyp[base - 1];
                    }
                }
                else
                {

                    if (base == 0)
                    {
                        if (propose > 0)
                        {
                            ret += propose_hyp_T[propose - 1];
                            if (propose > 1)
                            {
                                ret -= propose_hyp_T[propose - 2];
                            }
                        }
                    }

                    else
                    {
                        if (propose == 0)
                        {
                            ret = ret - base_hyp_T[base - 1] + base_hyp_T[base - 2];
                        }
                        else if (propose == 1)
                        {
                            ret += base_hyp_T[base - 2];
                        }
                    }
                }
            }
        }
        else
        {
            for (auto ch : l)
            {
                if (ch == side)
                {
                    propose++;
                }
                else if (ch == -side)
                {
                    base++;
                }
            }
            if (T_T)
            {
                if (propose != 0 && base == 0)
                {
                    ret += propose_hyp[propose - 1];
                }
                if (propose == 0 && base != 0)
                {
                    ret += base_hyp[base - 1];
                }
            }
            else
            {

                if (base == 0)
                {
                    if (propose > 0)
                    {
                        ret += propose_hyp_T[propose - 1];
                        if (propose > 1)
                        {
                            ret -= propose_hyp_T[propose - 2];
                        }
                    }
                }

                else
                {
                    if (propose == 0)
                    {
                        ret = ret - base_hyp_T[base - 1] + base_hyp_T[base - 2];
                    }
                    else if (propose == 1)
                    {
                        ret += base_hyp_T[base - 2];
                    }
                }
            }
        }
        return ret;
    }
    return 0;
}
int mcteva = 0;
int llt = 0;

void lee_testfour_mcts(vector<int> &l, int x, int y, int side, int diraction)
{

    if (x == 6 && y == 9 && diraction == 0)
    {
        int eee = 8;
    }
    if (mct_cout)
    {
        return;
    }
    if (l[1] != 1314 && l[6] != 1314)
    {
        if (l[0] == grid_blank && l[1] == side * -1 && l[2] == side * -1 && l[3] == side * -1 && l[4] == side * -1 && l[6] == grid_blank)
        {
            gridInfo_41[x][y]++;
            gridInfo_4_win = max(gridInfo_4_win, gridInfo_41[x][y]);
        }
        if (l[1] == side * -1 && l[2] == side * -1 && l[3] == side * -1 && l[0] == side * -1 && l[4] == grid_blank)
        {
            gridInfo_42[x][y]++;
            gridInfo_4_win2 = max(gridInfo_4_win2, gridInfo_42[x][y]);
        }
        if (l[1] == side * -1 && l[2] == side * -1 && l[3] == side * -1 && l[4] == side * -1 && l[0] == grid_blank && l[6] != grid_blank)
        {
            gridInfo_41[x][y]++;
            gridInfo_4_win = max(gridInfo_4_win, gridInfo_41[x][y]);
        }
        if (l[2] == side * -1 && l[3] == side * -1 && l[4] == side * -1 && l[6] == side * -1 && (l[1] == grid_blank || l[7] == grid_blank))
        {
            gridInfo_41[x][y]++;
            gridInfo_4_win = max(gridInfo_4_win, gridInfo_41[x][y]);
        }

        if (l[1] == side && l[2] == side && l[3] == side && l[4] == side)
        {
            if (l[0] == grid_blank)
            {
                mct_cout = 1;
                if (diraction == 0)
                    cout << x << ' ' << y << ' ' << x - 5 << ' ' << y << endl;
                if (diraction == 1)
                    cout << x << ' ' << y << ' ' << x << ' ' << y - 5 << endl;
                if (diraction == 2)
                    cout << x << ' ' << y << ' ' << x - 5 << ' ' << y + 5 << endl;
                if (diraction == 3)
                    cout << x << ' ' << y << ' ' << x - 5 << ' ' << y - 5 << endl;
                return;
            }
            if (l[6] == grid_blank)
            {
                mct_cout = 1;
                if (diraction == 0)
                    cout << x << ' ' << y << ' ' << x + 1 << ' ' << y << endl;
                if (diraction == 1)
                    cout << x << ' ' << y << ' ' << x << ' ' << y + 1 << endl;
                if (diraction == 2)
                    cout << x << ' ' << y << ' ' << x + 1 << ' ' << y - 1 << endl;
                if (diraction == 3)
                    cout << x << ' ' << y << ' ' << x + 1 << ' ' << y + 1 << endl;
                return;
            }
        }

        if (l[0] == side && l[1] == side && l[2] == side && l[3] == side && l[4] == side)
        {
            mct_cout5 = 1;
            mct_cout = 1;
            Xout1 = x;
            Yout1 = y;
            return;
        }
    }

    reverse(l.begin(), l.end());
    if (l[1] != 1314 && l[6] != 1314)
    {
        if (l[0] == grid_blank && l[1] == side * -1 && l[2] == side * -1 && l[3] == side * -1 && l[4] == side * -1 && l[6] == grid_blank)
        {
            gridInfo_41[x][y]++;
            gridInfo_4_win = max(gridInfo_4_win, gridInfo_41[x][y]);
        }
        if (l[1] == side * -1 && l[2] == side * -1 && l[3] == side * -1 && l[0] == side * -1 && l[4] == grid_blank)
        {
            gridInfo_42[x][y]++;
            gridInfo_4_win2 = max(gridInfo_4_win2, gridInfo_42[x][y]);
        }
        if (l[1] == side * -1 && l[2] == side * -1 && l[3] == side * -1 && l[4] == side * -1 && l[0] == grid_blank && l[6] != grid_blank)
        {
            gridInfo_41[x][y]++;
            gridInfo_4_win = max(gridInfo_4_win, gridInfo_41[x][y]);
        }
        if (l[2] == side * -1 && l[3] == side * -1 && l[4] == side * -1 && l[6] == side * -1 && (l[1] == grid_blank || l[7] == grid_blank))
        {
            gridInfo_41[x][y]++;
            gridInfo_4_win = max(gridInfo_4_win, gridInfo_41[x][y]);
        }

        if (l[1] == side && l[2] == side && l[3] == side && l[4] == side)
        {
            if (l[0] == grid_blank)
            {
                mct_cout = 1;
                if (diraction == 0)
                    cout << x << ' ' << y << ' ' << x + 5 << ' ' << y << endl;
                if (diraction == 1)
                    cout << x << ' ' << y << ' ' << x << ' ' << y + 5 << endl;
                if (diraction == 2)
                    cout << x << ' ' << y << ' ' << x + 5 << ' ' << y - 5 << endl;
                if (diraction == 3)
                    cout << x << ' ' << y << ' ' << x + 5 << ' ' << y + 5 << endl;
                return;
            }
            if (l[6] == grid_blank)
            {
                mct_cout = 1;
                if (diraction == 0)
                    cout << x << ' ' << y << ' ' << x - 1 << ' ' << y << endl;
                if (diraction == 1)
                    cout << x << ' ' << y << ' ' << x << ' ' << y - 1 << endl;
                if (diraction == 2)
                    cout << x << ' ' << y << ' ' << x - 1 << ' ' << y + 1 << endl;
                if (diraction == 3)
                    cout << x << ' ' << y << ' ' << x - 1 << ' ' << y - 1 << endl;
                return;
            }
        }

        if (l[0] == side && l[1] == side && l[2] == side && l[3] == side && l[4] == side)
        {
            mct_cout5 = 1;
            mct_cout = 1;
            Xout1 = x;
            Yout1 = y;
            return;
        }
    }

    return;
}
int AIDA_eva(int x, int y, int side = currBotColor, int T_T = 1)
{
    if (ABTriger_O == 1)
    {

        long long ret = 0;

        vector<int> l1, l2, l3, l4;
        int trigerAB = 0;
        if (T_T)
        {
            trigerAB = 5;
        }
        else
        {
            trigerAB = 5;
        }

        for (int i = 1; i <= trigerAB; ++i)
        {
            if (inMap(x, y + i))
            {
                l1.push_back(gridInfo[x][y + i]);
            }
            else
            {
                if (mcteva == 1)
                {
                    l1.push_back(1314);
                }
            }
            if (inMap(x, y - i))
            {
                l2.push_back(gridInfo[x][y - i]);
            }
            else
            {
                if (mcteva == 1)
                {
                    l2.push_back(1314);
                }
            }
            if (inMap(x + i, y))
            {
                l3.push_back(gridInfo[x + i][y]);
            }
            else
            {
                if (mcteva == 1)
                {
                    l3.push_back(1314);
                }
            }
            if (inMap(x - i, y))
            {
                l4.push_back(gridInfo[x - i][y]);
            }
            else
            {
                if (mcteva == 1)
                {
                    l4.push_back(1314);
                }
            }
        }
        reverse(l2.begin(), l2.end());
        l2.push_back(gridInfo[x][y]);
        l2.insert(l2.end(), l1.begin(), l1.end());
        if (mcteva == 0)
        {

            ret += eva_F(l2, side, T_T, trigerAB);
        }
        else
        {
            lee_testfour_mcts(l2, x, y, side, 1);
        }

        reverse(l4.begin(), l4.end());
        l4.push_back(gridInfo[x][y]);
        l4.insert(l4.end(), l3.begin(), l3.end());
        if (mcteva == 0)
        {

            ret += eva_F(l4, side, T_T, trigerAB);
        }
        else
        {
            lee_testfour_mcts(l4, x, y, side, 0);
        }

        l1.clear();
        l2.clear();
        l3.clear();
        l4.clear();

        for (int i = 1; i <= trigerAB; ++i)
        {
            if (inMap(x + i, y + i))
            {
                l1.push_back(gridInfo[x + i][y + i]);
            }
            else
            {
                if (mcteva == 1)
                {
                    l1.push_back(1314);
                }
            }
            if (inMap(x - i, y - i))
            {
                l2.push_back(gridInfo[x - i][y - i]);
            }
            else
            {
                if (mcteva == 1)
                {
                    l2.push_back(1314);
                }
            }
            if (inMap(x + i, y - i))
            {
                l3.push_back(gridInfo[x + i][y - i]);
            }
            else
            {
                if (mcteva == 1)
                {
                    l3.push_back(1314);
                }
            }
            if (inMap(x - i, y + i))
            {
                l4.push_back(gridInfo[x - i][y + i]);
            }
            else
            {
                if (mcteva == 1)
                {
                    l4.push_back(1314);
                }
            }
        }
        reverse(l2.begin(), l2.end());
        l2.push_back(gridInfo[x][y]);
        l2.insert(l2.end(), l1.begin(), l1.end());
        if (mcteva == 0)
        {
            ret += eva_F(l2, side, T_T, trigerAB);
        }
        else
        {
            lee_testfour_mcts(l2, x, y, side, 3);
        }

        reverse(l4.begin(), l4.end());
        l4.push_back(gridInfo[x][y]);
        l4.insert(l4.end(), l3.begin(), l3.end());
        if (mcteva == 0)
        {

            ret += eva_F(l4, side, T_T, trigerAB);
        }
        else
        {
            lee_testfour_mcts(l4, x, y, side, 2);
        }

        return ret;
    }
    return 0;
}

int eva_JA(int side)
{
    int nums = testPos.size();
    int r = 0;

    for (int i = 0; i < nums; ++i)
    {
        r = r + AIDA_eva(testPos[i].x, testPos[i].y, side, 0);
    }
    return r;
}

auto BeginTime = chrono::high_resolution_clock::now();
int Duration;
bool FRAMEWORK()
{
    auto RightNow = chrono::high_resolution_clock::now();
    Duration = chrono::duration_cast<chrono::milliseconds>(RightNow - BeginTime).count();

#if defined(_BOTZONE_ONLINE)

    return Duration >= 2980;
#else

    printf("Duration: %d\n", Duration);
    debugtime--;
    int reta = Duration >= 2980;
    printf("reta: %d\n", reta);
    return reta;
#endif
}

int ALandBE(int depth, int alpha, int beta, int side)
{
    llt++;
    if (FRAMEWORK())
    {
        return 0;
    }

    if (depth == 0)
    {
        int ret = eva_JA(side);
        return ret;
    }

    int nums = possiblePos.size();
    for (int i = 0; i < ZePHYr_One && i < nums; ++i)
    {
        if (gridInfo[possiblePos[i].x][possiblePos[i].y] != grid_blank)
        {
            continue;
        }
        testPos.push_back(possiblePos[i]);
        gridInfo[possiblePos[i].x][possiblePos[i].y] = side;

        for (int j = i + 1; j < ZePHYr_One && j < nums; ++j)
        {
            if (gridInfo[possiblePos[j].x][possiblePos[j].y] != grid_blank)
            {
                continue;
            }
            testPos.push_back(possiblePos[j]);
            gridInfo[possiblePos[j].x][possiblePos[j].y] = side;

            int cut = -ALandBE(depth - 1, -beta, -alpha, -side);
            gridInfo[possiblePos[j].x][possiblePos[j].y] = grid_blank;
            testPos.pop_back();

            if (cut > beta || cut == beta)
            {
                gridInfo[possiblePos[i].x][possiblePos[i].y] = grid_blank;
                testPos.pop_back();
                return beta;
            }
            if (cut > alpha && depth == 2)
            {
                Xout1 = testPos[0].x;
                Yout1 = testPos[0].y;
                Xout2 = possiblePos[j].x;
                Yout2 = possiblePos[j].y;
            }
            if (cut > alpha)
            {
                alpha = cut;
            }
        }
        gridInfo[possiblePos[i].x][possiblePos[i].y] = grid_blank;
        testPos.pop_back();
    }
    return alpha;
}

enum Play
{
    SELF = 1,
    FOE = -1,
    BLANK = 3

};

static int self_constant_blank[6][3] = {
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 3},
    {1, 3, 12},
    {1, 100, 10030},
    {1, 10080, 10080}};

static int self_blank_blank[6][3] =
    {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1},
        {1, 3, 5},
        {1, 115, 120},
        {900, 960, 1050}};

static int foe_constant_blank[6][3] = {
    {1, 1, 1},
    {1, 1, 1},
    {1, 1, 2},
    {1, 4, 10},
    {1, 110, 10100},
    {1, 10050, 10100}};

static int foe_blank_blank[6][3] =
    {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1},
        {1, 2, 4},
        {1, 110, 120},
        {900, 940, 1050}};

class State
{
public:
    enum left_or_right
    {
        left = 0,
        right = 1
    };

    Play player;

    int constant_len[2];

    int blank_len[2];

    bool is_blank[2];

    bool is_2_blank[2];

    void update(State &a, Play a_player, int direction)
    {
        if (a_player == BLANK)
        {
            constant_len[direction] = 0;
            blank_len[direction] = a.constant_len[direction] + 1;
            is_blank[direction] = true;
            is_2_blank[direction] = a.is_blank[direction];
        }
        else
        {
            if (player == a_player)
            {
                constant_len[direction] = a.constant_len[direction] + 1;
                blank_len[direction] = a.blank_len[direction] + 1;
                is_blank[direction] = a.is_blank[direction];
                is_2_blank[direction] = a.is_2_blank[direction];
            }
            else if (player == (a_player * -1))
            {
                constant_len[direction] = 0;
                blank_len[direction] = 0;
                is_blank[direction] = false;
                is_2_blank[direction] = false;
            }
        }
    }

    long long evaluate_road(Play a_player)
    {
        int total = 1 + constant_len[0] + constant_len[1];
        int total_is_blank = is_blank[0] + is_blank[1];
        if (total >= 6)
            return WIN;

        int left_total_blank = constant_len[1] + blank_len[0];
        int left_total_is_2_blank = is_blank[1] + is_2_blank[0];

        int right_total_blank = constant_len[0] + blank_len[1];
        int right_total_is_2_blank = is_blank[0] + is_2_blank[1];

        if (a_player == SELF)
        {
            long long value_num = self_constant_blank[total][total_is_blank];
            long long left_value_num = self_blank_blank[(left_total_blank >= 5) ? 5 : left_total_blank][left_total_is_2_blank];
            long long right_value_num = self_blank_blank[(right_total_blank >= 5) ? 5 : right_total_blank][right_total_is_2_blank];
            int max = value_num;
            if (max < left_value_num)
                max = left_value_num;
            if (max < right_value_num)
                max = right_value_num;
            return max;
        }
        else
        {
            long long value_num = foe_constant_blank[total][total_is_blank];
            long long left_value_num = foe_blank_blank[(left_total_blank >= 5) ? 5 : left_total_blank][left_total_is_2_blank];
            long long right_value_num = foe_blank_blank[(right_total_blank >= 5) ? 5 : right_total_blank][right_total_is_2_blank];
            int max = value_num;
            if (max < left_value_num)
                max = left_value_num;
            if (max < right_value_num)
                max = right_value_num;
            return max;
        }
    }

    State()
    {
        player = BLANK;
        constant_len[0] = constant_len[1] = 0;
        blank_len[0] = blank_len[1] = 0;
        is_blank[0] = is_blank[1] = false;
        is_2_blank[0] = is_2_blank[1] = false;
    }

    State(State &a)
    {
        player = a.player;
        constant_len[0] = a.constant_len[0];
        constant_len[1] = a.constant_len[1];
        blank_len[0] = a.blank_len[0];
        blank_len[1] = a.blank_len[1];
        is_blank[0] = a.is_blank[0];
        is_blank[1] = a.is_blank[1];
        is_2_blank[0] = a.is_2_blank[0];
        is_2_blank[1] = a.is_2_blank[1];
    }
};

class Board_data;
class Move
{
public:
    static const int center = 7;
    ProcLocation proclocation;
    long long value;

    Move(int x, int y, long long value = 0)
    {
        proclocation.x = x;
        proclocation.y = y;
        this->value = value;
    }
    Move(const ProcLocation &a, Board_data *board);
    bool operator<(const Move &other) const
    {
        if (value != other.value)
            return value > other.value;

        int distA = max(abs(proclocation.x - center), abs(proclocation.y - center));
        int distB = max(abs(other.proclocation.x - center), abs(other.proclocation.y - center));
        if (distA != distB)
            return distA < distB;

        if (proclocation.x != other.proclocation.x)
            return proclocation.x < other.proclocation.x;

        return proclocation.y < other.proclocation.y;
    }

    bool operator==(const Move &other) const
    {
        return proclocation.x == other.proclocation.x && proclocation.y == other.proclocation.y;
    }
};

#define LEVEL 12
class MCTSNode
{
public:
    int level;
    Play player;

    Play WINer;

    int Num_Visits;

    int Num_Reward;
    double NUM_UCB;

    Move *first;

    Move *second;

    MCTSNode *parent;
    vector<MCTSNode *> children;
    MCTSNode(int level, Play player, MCTSNode *parent = nullptr)
        : level(level), player(player), Num_Visits(0), Num_Reward(0), NUM_UCB(0), parent(parent), first(nullptr), second(nullptr), WINer(BLANK)
    {
    }
    void update_MCTSNode(Play winner)
    {
        Num_Visits += 2;
        if (winner == BLANK)
        {
            Num_Reward++;
            return;
        }
        if (winner == player * -1)
        {
            Num_Reward += 2;
        }
    }

    double UCB_cal()
    {
        NUM_UCB = (double)Num_Reward / Num_Visits + ucb_c * sqrt(log(parent->Num_Visits) / Num_Visits);
        return NUM_UCB;
    }

    double R_V = 0;
    double R_V_cal()
    {
        if (Num_Visits == 0)
        {
            R_V = 0;
        }
        else
        {
            R_V = (((1.0) * Num_Reward) / ((1.0) * Num_Visits));
        }
        return R_V;
    }
};

#define DIRECTION 4

int DIR[DIRECTION][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};

class Board_data
{
public:
    Play board_Condition[GRIDSIZE][GRIDSIZE];
    set<Move> Can_Move;

    State ProcLocation_state[GRIDSIZE][GRIDSIZE][DIRECTION][2];

    long long ProcLocation_value[GRIDSIZE][GRIDSIZE][2];

    long long get_ProcLocation_value(int x, int y, Play a_player)
    {
        int play_num = 0;
        if (a_player != SELF)
            play_num = 1;
        long long value = 1;
        for (int i = 0; i < DIRECTION; i++)
        {
            long long evalue = ProcLocation_state[x][y][i][play_num].evaluate_road(a_player);
            if (evalue == WIN)
                return WIN;
            else
                value = value * evalue;
        }
        return value;
    }

    auto &get_state(const ProcLocation &proclocation, const int &direction)
    {
        return ProcLocation_state[proclocation.x][proclocation.y][direction];
    }

    Board_data()
    {
        for (int i = 0; i < GRIDSIZE; i++)
        {
            for (int j = 0; j < GRIDSIZE; j++)
            {
                board_Condition[i][j] = BLANK;
                Move *x = new Move(i, j);
                for (int dir = 0; dir < DIRECTION; dir++)
                {
                    ProcLocation_state[i][j][dir][0].player = SELF;
                    ProcLocation_state[i][j][dir][1].player = FOE;
                    if (inMap(i + DIR[dir][0], j + DIR[dir][1]))
                    {
                        if (inMap(i + 2 * DIR[dir][0], j + 2 * DIR[dir][1]))
                        {
                            ProcLocation_state[i][j][dir][0].is_blank[1] = 1;
                            ProcLocation_state[i][j][dir][0].blank_len[1] = 1;
                            ProcLocation_state[i][j][dir][0].is_2_blank[1] = 1;
                            ProcLocation_state[i][j][dir][1].is_blank[1] = 1;
                            ProcLocation_state[i][j][dir][1].blank_len[1] = 1;
                            ProcLocation_state[i][j][dir][1].is_2_blank[1] = 1;
                        }
                        else
                        {
                            ProcLocation_state[i][j][dir][0].is_blank[1] = 1;
                            ProcLocation_state[i][j][dir][0].blank_len[1] = 1;
                            ProcLocation_state[i][j][dir][0].is_2_blank[1] = 0;
                            ProcLocation_state[i][j][dir][1].is_blank[1] = 1;
                            ProcLocation_state[i][j][dir][1].blank_len[1] = 1;
                            ProcLocation_state[i][j][dir][1].is_2_blank[1] = 0;
                        }
                    }
                    if (inMap(i - DIR[dir][0], j - DIR[dir][1]))
                    {
                        if (inMap(i - 2 * DIR[dir][0], j - 2 * DIR[dir][1]))
                        {
                            ProcLocation_state[i][j][dir][0].is_blank[0] = 1;
                            ProcLocation_state[i][j][dir][0].blank_len[0] = 1;
                            ProcLocation_state[i][j][dir][0].is_2_blank[0] = 1;
                            ProcLocation_state[i][j][dir][1].is_blank[0] = 1;
                            ProcLocation_state[i][j][dir][1].blank_len[0] = 1;
                            ProcLocation_state[i][j][dir][1].is_2_blank[0] = 1;
                        }
                        else
                        {
                            ProcLocation_state[i][j][dir][0].is_blank[0] = 1;
                            ProcLocation_state[i][j][dir][0].blank_len[0] = 1;
                            ProcLocation_state[i][j][dir][0].is_2_blank[0] = 0;
                            ProcLocation_state[i][j][dir][1].is_blank[0] = 1;
                            ProcLocation_state[i][j][dir][1].blank_len[0] = 1;
                            ProcLocation_state[i][j][dir][1].is_2_blank[0] = 0;
                        }
                    }
                }
                ProcLocation_value[i][j][0] = get_ProcLocation_value(i, j, SELF);
                ProcLocation_value[i][j][1] = get_ProcLocation_value(i, j, FOE);
                x->value = ProcLocation_value[i][j][0] > ProcLocation_value[i][j][1] ? ProcLocation_value[i][j][0] : ProcLocation_value[i][j][1];
                Can_Move.emplace(*x);
            }
        }
        int debug78 = 9;
    }

    void update_Board(ProcLocation &a, Play a_player)
    {
        int x = a.x;
        int y = a.y;
        if (x == -1 || !inMap(x, y))
            return;

        if (board_Condition[x][y] == a_player)
            return;

        if (board_Condition[x][y] == BLANK)
        {
            Can_Move.erase({a, this});
        }

        static int changed_ProcLocation[GRIDSIZE][GRIDSIZE] = {0};
        static int change_times = 0;
        vector<Move> changed_move_list;

        board_Condition[x][y] = a_player;
        change_times++;

        for (int dir = 0; dir < 4; dir++)
        {
            for (int p = 0; p < 2; p++)
            {
                const State &my_state = get_state(a, dir)[p];
                for (int lr = 0; lr < 2; lr++)
                {
                    const ProcLocation &Dir = lr ? ProcLocation(DIR[dir][0], DIR[dir][1]) : ProcLocation(DIR[dir][0] * -1, DIR[dir][1] * -1);
                    const ProcLocation &Target = a + (my_state.blank_len[lr] +
                                                      my_state.is_2_blank[lr] + 1) *
                                                         Dir;
                    ProcLocation tmp = a + Dir;
                    while (tmp != Target && inMap(tmp.x, tmp.y))
                    {
                        if (board_Condition[tmp.x][tmp.y] == BLANK &&
                            changed_ProcLocation[tmp.x][tmp.y] != change_times)
                        {
                            changed_ProcLocation[tmp.x][tmp.y] = change_times;
                            changed_move_list.emplace_back(tmp, this);
                        }
                        get_state(tmp, dir)[p].update(get_state(tmp - Dir, dir)[p],
                                                      board_Condition[(tmp - Dir).x][(tmp - Dir).y],
                                                      (State::left_or_right)(lr ^ 1));
                        tmp = tmp + Dir;
                    }
                }
            }
        }

        for (const Move &tmp : changed_move_list)
        {
            Can_Move.erase(tmp);
        }

        for (const Move &tmp : changed_move_list)
        {
            ProcLocation_value[tmp.proclocation.x][tmp.proclocation.y][0] = get_ProcLocation_value(tmp.proclocation.x, tmp.proclocation.y, SELF);
            ProcLocation_value[tmp.proclocation.x][tmp.proclocation.y][1] = get_ProcLocation_value(tmp.proclocation.x, tmp.proclocation.y, FOE);
            Can_Move.emplace(tmp.proclocation, this);
        }

        if (a_player == BLANK)
        {
            Can_Move.emplace(x, y, ProcLocation_value[x][y][0] > ProcLocation_value[x][y][1] ? ProcLocation_value[x][y][0] : ProcLocation_value[x][y][1]);
        }
    }
};

Move::Move(const ProcLocation &a, Board_data *board)
{
    proclocation.x = a.x;
    proclocation.y = a.y;
    value = board->ProcLocation_value[proclocation.x][proclocation.y][0] > board->ProcLocation_value[proclocation.x][proclocation.y][1] ? board->ProcLocation_value[proclocation.x][proclocation.y][0] : board->ProcLocation_value[proclocation.x][proclocation.y][1];
}

static constexpr int BREADTH[DEPTH + 1] = {2, 4, 4, 5, 5, 6, 6, 8, 8, 12, 12, 14, 16};
class MCTS
{
public:
    Board_data *board;

    void addmove(vector<Move> &moves, const int &depth, Play a_player)
    {
        const auto &x = board->Can_Move.begin();
        int m = (a_player == SELF) ? 0 : 1;
        const long long min_weight = ProcLocation_Value_Min > sqrt(board->ProcLocation_value[x->proclocation.x][x->proclocation.y][m]) ? sqrt(board->ProcLocation_value[x->proclocation.x][x->proclocation.y][m]) : ProcLocation_Value_Min;

        for (const Move &move : board->Can_Move)
        {
            if ((int)moves.size() >= (max(2, BREADTH[depth] / 2)) &&
                (board->ProcLocation_value[move.proclocation.x][move.proclocation.y][m] < min_weight || (int)moves.size() >= BREADTH[depth]))
                break;
            moves.emplace_back(move);
#if defined(_BOTZONE_ONLINE)
            ;
#else
            if (total_debug4 < 100)
            {
                cout << move.proclocation.x << " " << move.proclocation.y << endl;
            }
#endif
        }
        int debug99 = 0;
    }

    MCTSNode *Selection(MCTSNode &mctsnode)
    {
        MCTSNode *decision = nullptr;
        double NUM_UCB = 0.0;
        for (MCTSNode *ch : mctsnode.children)
        {
            if (ch->Num_Visits == 0)
            {
                decision = ch;
                break;
            }
            double chUCB = ch->UCB_cal();
            if (chUCB > NUM_UCB)
            {
                decision = ch;
                NUM_UCB = chUCB;
            }
            else if (chUCB == NUM_UCB)
            {
                if (decision->Num_Visits < ch->Num_Visits - 13)
                {
                    decision = ch;
                    NUM_UCB = chUCB;
                }
            }
        }
        return decision;
    }

    void Expansion(MCTSNode &mctsnode, Play &player)
    {
#if defined(_BOTZONE_ONLINE)
        ;
#else
        total_debug1++;
#endif
        int m;
        if (mctsnode.player == SELF)
            m = 0;
        else
            m = 1;
        if (mctsnode.WINer == BLANK && mctsnode.level != 0)
        {
            vector<Move> moves;
            addmove(moves, mctsnode.level, player);
            set<pair<ProcLocation, ProcLocation>> visited;
            for (int i = 0; i < moves.size(); i++)
            {
#if defined(_BOTZONE_ONLINE)
                ;
#else
                total_debug4++;
#endif
                Move &proclocation = moves[i];

                if (board->ProcLocation_value[proclocation.proclocation.x][proclocation.proclocation.y][m] == WIN)
                {
                    MCTSNode *winchildren = new MCTSNode(mctsnode.level - 1, (Play)(mctsnode.player * -1), &mctsnode);
                    winchildren->first = new Move(proclocation.proclocation.x, proclocation.proclocation.y, proclocation.value);
                    winchildren->second = new Move(moves[i == 0 ? 1 : i - 1].proclocation.x, moves[i == 0 ? 1 : i - 1].proclocation.y, moves[i == 0 ? 1 : i - 1].value);
                    winchildren->WINer = mctsnode.player;
                    mctsnode.children.push_back(winchildren);
                    break;
                }
                board->update_Board(proclocation.proclocation, mctsnode.player);

                bool win = 0;
                int count = 0;
                for (const Move &ch : board->Can_Move)
                {

                    if (visited.insert(make_pair(min(proclocation.proclocation, ch.proclocation), max(proclocation.proclocation, ch.proclocation))).second)
                    {
                        count++;
                        if (count > (BREADTH[mctsnode.level] - i) / 2 + 1)
                            break;

                        if (board->ProcLocation_value[ch.proclocation.x][ch.proclocation.y][m] == WIN)
                        {
#if defined(_BOTZONE_ONLINE)
                            ;
#else
                            total_debug2++;
#endif
                            win = 1;
                            MCTSNode *winchildren = new MCTSNode(mctsnode.level - 1, (Play)(mctsnode.player * -1), &mctsnode);
                            winchildren->first = new Move(proclocation.proclocation.x, proclocation.proclocation.y, proclocation.value);
                            winchildren->second = new Move(ch.proclocation.x, ch.proclocation.y, ch.value);
                            winchildren->WINer = mctsnode.player;
                            mctsnode.children.push_back(winchildren);
                            break;
                        }
                        else
                        {
#if defined(_BOTZONE_ONLINE)
                            ;
#else
                            total_debug3++;
#endif

                            MCTSNode *child = new MCTSNode(mctsnode.level - 1, (Play)(mctsnode.player * -1), &mctsnode);
                            child->first = new Move(proclocation.proclocation.x, proclocation.proclocation.y, proclocation.value);
                            child->second = new Move(ch.proclocation.x, ch.proclocation.y, ch.value);
                            mctsnode.children.push_back(child);
                        }
                    }
                }

                board->update_Board(proclocation.proclocation, BLANK);
                if (win)
                    break;
            }
        }
    }

    Play MCTS_main(MCTSNode &mctsnode)
    {
#if defined(_BOTZONE_ONLINE)
        ;
#else
        total_debug++;
#endif

        ucb_c = A * exp(-K * mctsnode.level);
        if (mctsnode.Num_Visits == 0)
        {
            Expansion(mctsnode, mctsnode.player);
        }
        if (mctsnode.children.empty())
        {
            mctsnode.update_MCTSNode(mctsnode.WINer);
            return mctsnode.WINer;
        }

        MCTSNode *decision = Selection(mctsnode);

        board->update_Board(decision->first->proclocation, mctsnode.player);
        board->update_Board(decision->second->proclocation, mctsnode.player);

        Play next_winer = MCTS_main(*decision);

        board->update_Board(decision->first->proclocation, BLANK);
        board->update_Board(decision->second->proclocation, BLANK);

        mctsnode.update_MCTSNode(next_winer);
        return next_winer;
    }
};

int JXout1 = -1;
int JYout1 = -1;
int JXout2 = -1;
int JYout2 = -1;
int chuan_JXout1()
{
    return JXout1;
}
int chuan_JYout1()
{
    return JYout1;
}
int chuan_JXout2()
{
    return JXout2;
}
int chuan_JYout2()
{
    return JYout2;
}

void AIDA_MCTS(vector<vector<int>> JGridInfo, int Jselffang)

{
    // int x0, y0, x1, y1;
    MCTS *mcts = new MCTS();
    mcts->board = new Board_data();
    /**
    for (int i = 0; i < turnID - 1; i++)
    {

        cin >> x0 >> y0 >> x1 >> y1;
        ProcLocation a(x0, y0);
        mcts->board->update_Board(a, FOE);
        ProcStep(x0, y0, x1, y1, -currBotColor, 0);
        ProcLocation a1(x1, y1);
        mcts->board->update_Board(a1, FOE);
        if (i == turnID - 1)
            break;
        cin >> x0 >> y0 >> x1 >> y1;
        ProcLocation a2(x0, y0);
        mcts->board->update_Board(a2, SELF);
        ProcStep(x0, y0, x1, y1, currBotColor, 0);
        ProcLocation a3(x1, y1);
        mcts->board->update_Board(a3, SELF);
    }
    cin >> x0 >> y0 >> x1 >> y1;
    ProcStep(x0, y0, x1, y1, -currBotColor, 0);
    ProcLocation a(x0, y0);
    mcts->board->update_Board(a, FOE);

    ProcLocation a1(x1, y1);
    mcts->board->update_Board(a1, FOE);
    ****/

    for (int i = 0; i < GRIDSIZE; i++)
    {
        for (int j = 0; j < GRIDSIZE; j++)
        {
            gridInfo[i][j] = JGridInfo[i][j];
            if (JGridInfo[i][j] == 0)
                continue;
            else
            {
                if (JGridInfo[i][j] == Jselffang)
                {
                    ProcLocation a(i, j);
                    mcts->board->update_Board(a, SELF);
                }
                else
                {
                    ProcLocation a(i, j);
                    mcts->board->update_Board(a, FOE);
                }
            }
        }
    }

    MCTSNode *root = new MCTSNode(12, SELF);
    mcteva = 1;
    if (mcteva)
    {
        for (int x = 0; x < GRIDSIZE; ++x)
        {
            for (int y = 0; y < GRIDSIZE; ++y)
            {
                if (gridInfo[x][y] == grid_blank)
                {

                    int val = AIDA_eva(x, y);
                }
            }
        }
        int yy = 9;
        if (mct_cout && mct_cout5 == 0)
        {
            return;
        }
        if (mct_cout && mct_cout5)
        {
            for (int x = 0; x < GRIDSIZE; ++x)
            {
                for (int y = 0; y < GRIDSIZE; ++y)
                {
                    if (gridInfo[x][y] == grid_blank)
                    {
                        // cout << Xout1 << ' ' << Yout1 << ' ' << x << ' ' << y << endl;
                        JXout1 = Xout1;
                        JYout1 = Yout1;
                        JXout2 = x;
                        JYout2 = y;
                        return;
                    }
                }
            }
        }
        lock_mct = 1;
        if (lock_mct)
        {
            int ok1 = 0, ok2 = 0;
            if (gridInfo_4_win >= 1)
            {
                for (int x = 0; x < GRIDSIZE; ++x)
                {
                    for (int y = 0; y < GRIDSIZE; ++y)
                    {
                        if (x == 6 && y == 9)
                        {
                            int eee = 8;
                        }
                        if (gridInfo[x][y] == grid_blank && gridInfo_41[x][y] == gridInfo_4_win)
                        {
                            Xout1 = x;
                            Yout1 = y;
                            ok1 = 1;
                            break;
                        }
                    }
                    if (ok1)
                    {
                        break;
                    }
                }
            }
            if (gridInfo_4_win >= 1)
            {
                for (int x = 0; x < GRIDSIZE; ++x)
                {
                    for (int y = 0; y < GRIDSIZE; ++y)
                    {
                        if (gridInfo_4_win2)
                        {
                            if (gridInfo_42[x][y] == gridInfo_4_win2)
                            {
                                int distA = max(abs(x - Xout1), abs(y - Yout1));
                                if (distA < 4)
                                    continue;
                                Xout2 = x;
                                Yout2 = y;
                                ok2 = 1;
                                break;
                            }
                        }
                        else if (gridInfo_41[x][y])
                        {
                            int distA = max(abs(x - Xout1), abs(y - Yout1));
                            if (distA < 4)
                                continue;
                            Xout2 = x;
                            Yout2 = y;
                            ok2 = 1;
                            break;
                        }
                    }
                    if (ok2)
                    {
                        break;
                    }
                }
            }

            if (ok1 && ok2 && inMap(Xout1, Yout1) && inMap(Xout2, Yout2))
            {
                // cout << Xout1 << ' ' << Yout1 << ' ' << Xout2 << ' ' << Yout2 << endl;
                JXout1 = Xout1;
                JYout1 = Yout1;
                JXout2 = Xout2;
                JYout2 = Yout2;
                return;
            }
        }
    }
	cout << "MCT" << endl;
    int dd = 9;
    while (!FRAMEWORK())
    {
        mcts->MCTS_main(*root);
#if defined(_BOTZONE_ONLINE)
        ;
#else
        cout << endl;
        cout << "totoal_debug:" << total_debug << endl;
        cout << "totoal_debug1:" << total_debug1 << endl;
        cout << "totoal_debug2:" << total_debug2 << endl;
        cout << "totoal_debug3:" << total_debug3 << endl;
        cout << "totoal_debug4:" << total_debug4 << endl;
        cout << "totoal_debug5:" << total_debug5 << endl;
        cout << "totoal_debug6:" << total_debug6 << endl;
#endif
    }

    double max_R_V = 0;
    MCTSNode *best = nullptr;
    for (MCTSNode *ch : root->children)
    {
        ch->R_V_cal();
        if (ch->Num_Visits > max_R_V)
        {
            max_R_V = ch->Num_Visits;
            best = ch;
        }

#if defined(_BOTZONE_ONLINE)
        ;
#else
        cout << ch->first->proclocation.x << " " << ch->first->proclocation.y << "  " << ch->second->proclocation.x << " " << ch->second->proclocation.y << "  NUM_VI:" << ch->Num_Visits << "  NUM_Re " << ch->Num_Reward << "  ucb:" << ch->NUM_UCB << "  R_V:" << ch->R_V << endl;
#endif
    }

#if defined(_BOTZONE_ONLINE)
    ;
#else
    cout << "max : " << max_R_V << endl;
#endif
    Xout1 = best->first->proclocation.x;
    Yout1 = best->first->proclocation.y;
    Xout2 = best->second->proclocation.x;
    Yout2 = best->second->proclocation.y;
    int sdwww = 9;
    // cout << Xout1 << ' ' << Yout1 << ' ' << Xout2 << ' ' << Yout2 << endl;
    JXout1 = Xout1;
    JYout1 = Yout1;
    JXout2 = Xout2;
    JYout2 = Yout2;
    return;
}

void main1(vector<vector<int>> JGridInfo, int JturnID, int Jselffang)
{
    int x0, y0, x1, y1;

    int turnID = JturnID;

    BeginTime = chrono::high_resolution_clock::now();
    currBotColor = Jselffang;
    int Leetriger = 0;

    if (turnID > 6)
    {
        AIDA_MCTS(JGridInfo, Jselffang);
        return;
    }
#if defined(_BOTZONE_ONLINE)
    return;
#else
    system("pause");
    return;
#endif
}

void AIDA_INIT(int turnID, int xxx0 = 10, int yyy0 = 10)
{
    int x0=xxx0, y0=yyy0, x1, y1;

    for (int x = 0; x < GRIDSIZE; ++x)
    {
        for (int y = 0; y < GRIDSIZE; ++y)
        {
            if (gridInfo[x][y] == grid_blank)
            {

                int val = AIDA_eva(x, y);

                ProcLocation oneT(x, y, val);

                possiblePos.push_back(oneT);
            }
        }
    }
#if defined(_BOTZONE_ONLINE)
    ;
#else
    cout << "Duration" << Duration << endl;
#endif

    sort(possiblePos.begin(), possiblePos.end(), cmp);
    Xout1 = possiblePos[0].x;
    Yout1 = possiblePos[0].y;
    Xout2 = possiblePos[1].x;
    Yout2 = possiblePos[1].y;
    if (currBotColor != grid_white)
    {
        //cout << 10 << ' ' << 8 << ' ' << -1 << ' ' << -1 << endl;
        Xout1 = 10;
        Yout1 =8;
    }
    else
    {
        int xx0 = x0;
        int yy0 = y0 - 2;
        int xx1 = x0 + 1;
        int yy1 = y0 + 1;
        if (inMap(xx0, yy0) && inMap(xx1, yy1))
        {
            //cout << xx0 << ' ' << yy0 << ' ' << xx1 << ' ' << yy1 << endl;
            Xout1 = xx0;
            Yout1 = yy0;
            Xout2 = xx1;
            Yout2 = yy1;
            return;
        }
        xx0 = x0;
        yy0 = y0 + 2;
        xx1 = x0 - 1;
        yy1 = y0 - 1;
        if (inMap(xx0, yy0) && inMap(xx1, yy1))
        {
            //cout << xx0 << ' ' << yy0 << ' ' << xx1 << ' ' << yy1 << endl;
            Xout1 = xx0;
            Yout1 = yy0;
            Xout2 = xx1;
            Yout2 = yy1;
            return;
        }
        ALandBE(2, INT_MIN + 100, INT_MAX - 100, currBotColor);
    }

    //cout << Xout1 << ' ' << Yout1 << ' ' << Xout2 << ' ' << Yout2 << endl;
    return;
}

void AIDA_AB(int turnID)
{
    int x0, y0, x1, y1;
   
    for (int x = 0; x < GRIDSIZE; ++x)
    {
        for (int y = 0; y < GRIDSIZE; ++y)
        {
            if (gridInfo[x][y] == grid_blank)
            {

                int val = AIDA_eva(x, y);
                ProcLocation oneT(x, y, val);

                possiblePos.push_back(oneT);
            }
        }
    }
#if defined(_BOTZONE_ONLINE)
    ;
#else
    cout << "Duration" << Duration << endl;
#endif

    sort(possiblePos.begin(), possiblePos.end(), cmp);
    Xout1 = possiblePos[0].x;
    Yout1 = possiblePos[0].y;
    Xout2 = possiblePos[1].x;
    Yout2 = possiblePos[1].y;
    if (turnID != 1 || currBotColor == grid_white)
    {
        ALandBE(2, INT_MIN + 100, INT_MAX - 100, currBotColor);
    }
    else
    {
        //cout << 7 << ' ' << 7 << ' ' << -1 << ' ' << -1 << endl;
        return;
    }

    //cout << Xout1 << ' ' << Yout1 << ' ' << Xout2 << ' ' << Yout2 << endl;
    return;
}

void mainAB(int &QXout1, int &QYout1, int &QXout2, int &QYout2, int turnID, int cur, vector<vector<int>> gri, int xxx0=10 , int yyy0=10 )
{
    int x0=xxx0, y0=yyy0, x1, y1;
    for (int i = 0; i < GRIDSIZE; i++)
    {
        for (int j = 0; j < GRIDSIZE; j++)
        {
            gridInfo[i][j] = gri[i][j];
        }
    }
    BeginTime = chrono::high_resolution_clock::now();
    currBotColor = cur;
    int Leetriger = 0;

    if (turnID <= 1)
    {
        AIDA_INIT(turnID,xxx0,yyy0);
        QXout1 = Xout1;
        QYout1 = Yout1;
        QXout2 = Xout2;
        QYout2 = Yout2;
        return ;
    }
    if (turnID <= 6 && turnID > 1)
    {
        AIDA_AB(turnID);
        QXout1 = Xout1;
        QYout1 = Yout1;
        QXout2 = Xout2;
        QYout2 = Yout2;
        return ;
    }

#if defined(_BOTZONE_ONLINE)
    return ;
#else
    system("pause");
    return ;
#endif
}



