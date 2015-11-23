#include <iostream>
#include "Matrix.h"

using namespace std;

int main(int argc, char** argv)
{
    cout<<"\nThis is a program to demonstrate using the CMatrix class for\n"
    <<"principal component analysis(PCA)"<<"\n\n"<<endl;    
    
    int m = 13;
    int n = 4;
    
    int i, j, k;
    
    //the ingredients data in the Hald data set
    float ingredients[13][4] = {{ 7,26, 6,60},
                               { 1,29,15,52},
                               {11,56, 8,20},
                               {11,31, 8,47},
                               { 7,52, 6,33},
                               {11,55, 9,22},
                               { 3,71,17, 6},
                               { 1,31,22,44},
                               { 2,54,18,22},
                               {21,47, 4,26},
                               { 1,40,23,34},
                               {11,66, 9,12},
                               {10,68, 8,12}};

    //print the data matrix
    cout<<"The data matrix:\n(rows correspond to observations, columns to variables)"<<endl;
    for(i = 0; i < m; i++)
    {
        for(j = 0; j < n; j++)
        {
            cout<<ingredients[i][j]<<" ";
        }
        cout<<endl;           
    } 
    cout<<endl;                          

    float** data = new float*[n];
    for(i = 0; i < n; i++)
    {
        data[i] = new float[m];
    }
    
    CMatrix mat;    
    float** cov_mat = mat.allocMat(n);
    
    //calculate the covariance matrix
    float* mean = new float[n];
    for(j = 0; j < n; j++)
    {   
        mean[j] = 0.0;
        for(i = 0; i < m; i++)
        {
            mean[j] += ingredients[i][j];
        }
        mean[j] /= m;
    }
    
    for(i = 0; i < m; i++)
    {
        for(j = 0; j < n; j++)
        {
            ingredients[i][j] -= mean[j];
            data[j][i] = ingredients[i][j];
        }
    }
    
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            cov_mat[i][j] = 0.0;
            for(k = 0; k < m; k++)
            {
                cov_mat[i][j] += data[i][k]*ingredients[k][j];
            }
            cov_mat[i][j] /= m-1;
        }
    }
    
    int vecNum = n; //number of principal eigenvectors(<=n)
    float* phi = new float[n*vecNum];
    float* lambda = new float[vecNum];
    
    //principal component analysis
    mat.PCA(cov_mat, n, phi, lambda, vecNum);
    
    //print the PCA result
    cout<<"The eigenvectors:\n(each row as an eigenvector)"<<endl;
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            cout<<phi[i*vecNum+j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
    
    //release the memory
    delete[] phi;
    delete[] lambda;
    mat.freeMat(cov_mat, n);
    delete mean;
    for(i = 0; i < n; i++)
    {
        delete []data[i];
        data[i] = NULL;
    }
    delete []data;
    data = NULL;

    return 0;
}
