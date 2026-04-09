#include <iostream>
#include <stdlib.h>
#include <vector>
//#include "AIDA.h"
#include "AIDA2.h"
using namespace std;
#define Qgrid_black 1
#define Qgrid_white -1
#define QGRIDSIZE 19
vector<vector<int>> QGridInfo(QGRIDSIZE, vector<int>(QGRIDSIZE, 0));
int Qselffang = 0;
int turnID = 0;
// 下面的应该为-1
int QXout1 = -1;
int QYout1 = -1;
int QXout2 = -1;
int QYout2 = -1;
int QXin1 = -1;
int QYin1 = -1;
int QXin2 = -1;
int QYin2 = -1;

// 重置棋盘
void reset_board()
{
    for (int i = 0; i < QGRIDSIZE; i++)
        for (int j = 0; j < QGRIDSIZE; j++)
            QGridInfo[i][j] = 0;
    Qselffang = 0;
    turnID = 0;
}

void kong() {};

// 开始游戏的函数
void play()
{
    while (1)
    {
        cout << "输入1为先手，-1为后手" << endl;
        cin >> Qselffang;
        turnID = 1;
        if (Qselffang == 1)
        {
            if (turnID == 1)
            {
                mainAB(QXout1, QYout1, QXout2, QYout2, turnID, Qselffang, QGridInfo);
                cout << QXout1 + 1 << " " << QYout1 + 1 << endl;
                QGridInfo[QXout1][QYout1] = Qgrid_black;
            }
            while (1)
            {
                cout << "请输入对方行棋：4个整数" << endl;
                cin >> QXin1 >> QYin1 >> QXin2 >> QYin2;
                QGridInfo[QXin1 - 1][QYin1 - 1] = Qgrid_white;
                QGridInfo[QXin2 - 1][QYin2 - 1] = Qgrid_white;
                turnID++;
                if (turnID <= 6 && turnID > 1)
                {
                    mainAB(QXout1, QYout1, QXout2, QYout2, turnID, Qselffang, QGridInfo);
                }
                else
                {
                    main1(QGridInfo, turnID, Qselffang);
                    QXout1 = chuan_JXout1();
                    QYout1 = chuan_JYout1();
                    QXout2 = chuan_JXout2();
                    QYout2 = chuan_JYout2();
                    //cout << QXout1 + 1 << ' ' << QYout1 + 1 << ' ' << QXout2 + 1 << ' ' << QYout2 + 1 << endl;
                }
                cout << QXout1 + 1 << ' ' << QYout1 + 1 << ' ' << QXout2 + 1 << ' ' << QYout2 + 1 << endl;

                QGridInfo[QXout1][QYout1] = Qgrid_black;
                QGridInfo[QXout2][QYout2] = Qgrid_black;
            }
        }
        if (Qselffang == -1)
        {
            if (turnID == 1)
            {
                cout << "请输入对方行棋：2个整数" << endl;
                cin >> QXin1 >> QYin1;
                QGridInfo[QXin1 - 1][QYin1 - 1] = Qgrid_black;

                mainAB(QXout1, QYout1, QXout2, QYout2, turnID, Qselffang, QGridInfo,QXin1 - 1,QYin1 - 1);
                cout << QXout1 + 1 << ' ' << QYout1 + 1 << ' ' << QXout2 + 1 << ' ' << QYout2 + 1 << endl;
                QGridInfo[QXout1][QYout1] = Qgrid_white;
                QGridInfo[QXout2][QYout2] = Qgrid_white;
            }
            while (1)
            {
                cout << "请输入对方行棋：4个整数" << endl;
                cin >> QXin1 >> QYin1 >> QXin2 >> QYin2;
                QGridInfo[QXin1 - 1][QYin1 - 1] = Qgrid_black;
                QGridInfo[QXin2 - 1][QYin2 - 1] = Qgrid_black;
                turnID++;

                if (turnID <= 6 && turnID > 1)
                {
                    mainAB(QXout1, QYout1, QXout2, QYout2, turnID, Qselffang, QGridInfo);
                }
                else
                {
                    main1(QGridInfo, turnID, Qselffang);
                    QXout1 = chuan_JXout1();
                    QYout1 = chuan_JYout1();
                    QXout2 = chuan_JXout2();
                    QYout2 = chuan_JYout2();
                    // cout << QXout1 + 1 << ' ' << QYout1 + 1 << ' ' << QXout2 + 1 << ' ' << QYout2 + 1 << endl;
                }
                cout << QXout1 + 1 << ' ' << QYout1 + 1 << ' ' << QXout2 + 1 << ' ' << QYout2 + 1 << endl;
                QGridInfo[QXout1][QYout1] = Qgrid_white;
                QGridInfo[QXout2][QYout2] = Qgrid_white;
            }
        }
    
    }
}

int main()
{

    system("title A.I.D.A");
    system("color F5");
    system("cls");
    play();
    reset_board();
}