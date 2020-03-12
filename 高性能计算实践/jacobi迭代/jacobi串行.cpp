#include <stdio.h>   
#include <stdlib.h>   
#include <sys/time.h>   
#include <ctime>   
using namespace std;


#define MN 900  //dimension of matrix
#define CNT 2000  //number of iteration  


 
double mat[MN][MN];   
       
int main()   
{
    srand((unsigned)time(NULL));   
       
    for(int i=0; i< MN; i++)   
    {   
        mat[i][0] = 0;   
        mat[0][i] = 0;   
        mat[i][MN-1] = 0;   
        mat[MN-1][i] = 0;   
    }   
       
    for(int i=1; i< MN-1; i++)   
        for(int j=1; j< MN-1; j++)   
            mat[i][j] = rand()%10;   
       
    double time;   
    struct timeval begin,end;   
    gettimeofday(&begin,NULL);   
    int div;   
       
    for(int k=0; k<CNT; k++)   
    {   
        for(int i=1; i< MN-1; i++)   
            for(int j=1; j<MN-1; j++)   
            {   
                div = 4;   
                if(i==1 || i==MN-2) div--;   
                if(j==1 || j==MN-2) div--;   
                mat[i][j] = (mat[i-1][j]+mat[i+1][j]+mat[i][j-1]+mat[i][j+1])/div;   
            }   
    }    
    gettimeofday(&end,NULL);   
    double s= end.tv_sec - begin.tv_sec;   
    double us=end.tv_usec - begin.tv_usec;   
    time =  s+us / 1000000;   
    printf("time is %.10lf s\n",time);   
}  