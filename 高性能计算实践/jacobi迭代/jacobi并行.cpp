#include <iostream>
#include <pthread.h>
#include <sys/time.h>
#include <cstdlib>
using namespace std;


#define num_Thread 4  //number of threads
#define Order 100  //order of matrix
#define Iteration 2000  //number of iteration


double mat[Order][Order];  //inite empty matrix

bool init()
{   
    srand((unsigned)time(NULL));
    for(int i = 0; i < Order; i++)
    {   
        mat[i][0] = 0;
        mat[0][i] = 0;
        mat[i][Order-1] = 0;
        mat[Order-1][i] = 0;
    }
   
    for(int i = 1; i < Order-1; i++)   
        for(int j = 1; j < Order-1; j++)   
            mat[i][j] = rand()%10;
    return true;   
}
 
void* Jacobi_parallel(void *id)   
{   
    int tid =*(int*)id;   
    int each = (Order - 2) / num_Thread;   
    int f = tid*each+1;   
    int t = f + each;   
   
    for(int k = 0 ; k < Iteration; k++)   
        for(int i = f; i < t; i++)   
        {   
            for(int j = 1; j < Order-1; j++)   
            {   
                int div = 4;   
                if(i==1 || i==Order-2) 
					div--;   
                if(j==1 || j==Order-2) 
					div--;   
                mat[i][j] = (mat[i-1][j] + mat[i+1][j] + mat[i][j-1] + mat[i][j+1]) / div;   
            }   
        }   
}   
  
int main()   
{   
    double time;   
    struct timeval begin, end;   
    gettimeofday(&begin, NULL);   
    
    if(!init()) return 0;
    pthread_t Thread_ids[num_Thread];   
   
    int error;   
   
    int ids[num_Thread];   
    for(int i = 0; i < num_Thread; i++)   
    {   
        ids[i] = i;   
        pthread_create(&Thread_ids[i], NULL, Jacobi_parallel, &ids[i]);   
    }   
   
    for(int i = 0; i < num_Thread; i++)   
    {   
        error = pthread_join(Thread_ids[i], NULL);   
        if(error)   
        {   
            cout<<"thread "<< i <<" error"<<endl;   
            exit(0);   
        }
    }
	    gettimeofday(&end, NULL);   
        double s = end.tv_sec - begin.tv_sec;   
        double us = end.tv_usec - begin.tv_usec;   
        time =  s + us / 1000000;
        printf("time is %.10lf s\n",time);
    return 0;
}
